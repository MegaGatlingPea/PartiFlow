"""OpenMM worker resource reuse through the public topology relax function."""

from __future__ import annotations

from contextlib import ExitStack
import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace as NS
import unittest
from unittest.mock import Mock, call, patch


ROOT = Path(__file__).resolve().parents[1]


class Residue:
    def __init__(self, name="ALA", number="1"):
        self.name = name
        self.id = number
        self.chain = None
        self._atoms = [
            NS(name="CA", residue=self, element=NS(name="carbon")),
            NS(name="H", residue=self, element=NS(name="hydrogen")),
        ]

    def atoms(self):
        return iter(self._atoms)


class Topology:
    def __init__(self):
        self.residue = Residue()
        chain = NS(id="L", residues=lambda: iter([self.residue]))
        self.residue.chain = chain
        self._chain = chain

    def chains(self):
        return iter([self._chain])

    def residues(self):
        return iter([self.residue])

    def atoms(self):
        return self.residue.atoms()

    def bonds(self):
        return iter(())


class OpenMMRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.forcefields = []
        self.systems = []
        self.forces = []
        self.simulations = []

        class System:
            def __init__(self):
                self.forces = []

            def addForce(self, force):
                self.forces.append(force)

        owner = self

        class ForceField:
            def __init__(self, *files):
                self.files = files
                self.calls = []
                owner.forcefields.append(self)

            def createSystem(self, topology, **options):
                system = System()
                self.calls.append((topology, options, system))
                owner.systems.append(system)
                return system

        class Force:
            def __init__(self, expression):
                self.expression = expression
                self.global_parameters = {}
                self.particle_parameters = []
                self.particles = []
                owner.forces.append(self)

            def addGlobalParameter(self, name, value):
                self.global_parameters[name] = value

            def addPerParticleParameter(self, name):
                self.particle_parameters.append(name)

            def addParticle(self, index, positions):
                self.particles.append((index, positions))

        class Context:
            def __init__(self):
                self.positions = None
                self.state_requests = []

            def setPositions(self, positions):
                self.positions = positions

            def getState(self, **options):
                self.state_requests.append(options)
                energy = 10.0 if len(self.state_requests) == 1 else 5.0
                return NS(
                    getPotentialEnergy=lambda: NS(value_in_unit=lambda _: energy),
                    getPositions=lambda: self.positions,
                )

        class Simulation:
            def __init__(self, topology, system, integrator, platform,
                         *, platformProperties):
                self.topology = topology
                self.system = system
                self.integrator = integrator
                self.platform = platform
                self.properties = dict(platformProperties)
                self.context = Context()
                self.minimizeEnergy = Mock()
                owner.simulations.append(self)

        app = ModuleType("openmm.app")
        app.ForceField = Mock(side_effect=ForceField)
        app.HBonds = object()
        app.Simulation = Mock(side_effect=Simulation)

        unit = ModuleType("openmm.unit")
        for name in ("kilocalories_per_mole", "angstroms", "kilojoules_per_mole",
                     "kilojoule_per_mole", "nanometer"):
            setattr(unit, name, 1.0)

        openmm = ModuleType("openmm")
        openmm.app = app
        openmm.unit = unit
        openmm.CustomExternalForce = Mock(side_effect=Force)
        openmm.LangevinIntegrator = Mock(side_effect=lambda *args: NS(args=args))
        openmm.Platform = NS(
            getPlatformByName=Mock(side_effect=lambda name: NS(name=name))
        )

        topology_module = ModuleType("partiflow.postprocess.relax.topology")
        topology_module._write_pdb = Mock(return_value="HEADER TEST\nEND\n")
        topology_module._add_connects = Mock(side_effect=lambda text, _: text)

        self.stack.enter_context(patch.dict(sys.modules, {
            "openmm": openmm,
            "openmm.app": app,
            "openmm.unit": unit,
            "pdbfixer": ModuleType("pdbfixer"),
            "partiflow.postprocess.relax.topology": topology_module,
        }))
        module_path = ROOT / "partiflow/postprocess/relax/openmm_relaxer.py"
        spec = importlib.util.spec_from_file_location(
            "postprocess_openmm_runtime_tests", module_path
        )
        self.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.module)
        self.filtered_xml = self.stack.enter_context(
            patch.object(
                self.module, "_build_filtered_forcefield_xml",
                side_effect=lambda names: "custom-" + "-".join(names) + ".xml",
            )
        )
        self.app = app
        self.openmm = openmm
        self.topology_module = topology_module

    @staticmethod
    def fixed(*names, overrides=None, offset=0.0):
        return NS(
            topology=Topology(),
            positions=[(offset, 0.0, 0.0), (offset, 1.0, 0.0)],
            custom_residue_names=set(names),
            residue_template_overrides=overrides or {},
            connects=[],
        )

    def run_relax(self, fixed=None, **options):
        return self.module.run_topology_relax(
            fixed or self.fixed(), platform=options.pop("platform", "CPU"),
            **options,
        )

    def test_same_residue_set_reuses_one_forcefield_in_sorted_order(self):
        self.run_relax(self.fixed("NLE", "MK8"))
        self.run_relax(self.fixed("MK8", "NLE"))
        self.app.ForceField.assert_called_once_with(
            "amber14-all.xml", "custom-MK8-NLE.xml"
        )
        self.filtered_xml.assert_called_once_with(("MK8", "NLE"))
        self.assertEqual(len(self.forcefields[0].calls), 2)

    def test_distinct_residue_sets_have_separate_forcefields(self):
        for names in ((), ("MK8",), ("NLE",), ("MK8",), ()):
            self.run_relax(self.fixed(*names))
        self.assertEqual(self.app.ForceField.call_args_list, [
            call("amber14-all.xml"),
            call("amber14-all.xml", "custom-MK8.xml"),
            call("amber14-all.xml", "custom-NLE.xml"),
        ])
        self.assertEqual(
            [len(forcefield.calls) for forcefield in self.forcefields], [2, 2, 1]
        )
        self.assertEqual(self.filtered_xml.call_args_list, [
            call(("MK8",)), call(("NLE",))
        ])

    def test_template_overrides_are_part_of_forcefield_key(self):
        first = self.fixed(overrides={("L", "1"): "NMK8"})
        second = self.fixed(overrides={("L", "1"): "NNLE"})
        self.run_relax(first)
        self.run_relax(second)
        self.assertEqual(self.filtered_xml.call_args_list, [
            call(("NMK8",)), call(("NNLE",))
        ])
        self.assertEqual(
            self.forcefields[0].calls[0][1]["residueTemplates"],
            {first.topology.residue: "NMK8"},
        )
        self.assertEqual(first.custom_residue_names, set())

    def test_each_sample_has_its_own_system_simulation_context_and_restraints(self):
        first, second = self.fixed("MK8", offset=1.0), self.fixed("MK8", offset=9.0)
        text, energy = self.run_relax(first, stiffness=7, max_iterations=11)
        self.run_relax(second, stiffness=3, max_iterations=22)
        self.assertEqual(len(self.forcefields), 1)
        self.assertEqual(len(self.systems), 2)
        self.assertIsNot(self.systems[0], self.systems[1])
        left, right = self.simulations
        self.assertIsNot(left, right)
        self.assertIsNot(left.context, right.context)
        self.assertIsNot(left.integrator, right.integrator)
        self.assertIs(left.context.positions, first.positions)
        self.assertIs(right.context.positions, second.positions)
        self.assertIs(left.system, self.systems[0])
        self.assertIs(right.system, self.systems[1])
        self.assertEqual([len(system.forces) for system in self.systems], [1, 1])
        self.assertIsNot(self.forces[0], self.forces[1])
        self.assertEqual(self.forces[0].global_parameters, {"k": 7.0})
        self.assertEqual(self.forces[1].global_parameters, {"k": 3.0})
        self.assertEqual(self.forces[0].particles, [(0, first.positions[0])])
        self.assertEqual(self.forces[1].particles, [(0, second.positions[0])])
        left.minimizeEnergy.assert_called_once_with(maxIterations=11, tolerance=2.39)
        right.minimizeEnergy.assert_called_once_with(maxIterations=22, tolerance=2.39)
        self.assertEqual(energy, {"einit": 10.0, "efinal": 5.0})
        self.assertIn("INITIAL ENERGY: 10.000", text)
        self.assertEqual(self.topology_module._write_pdb.call_args_list, [
            call(first.topology, first.positions),
            call(second.topology, second.positions),
        ])

    def test_requested_cpu_threads_reach_each_simulation(self):
        for threads in (1, 4):
            self.run_relax(threads=threads)
        self.assertEqual(
            [simulation.properties for simulation in self.simulations],
            [{"Threads": "1"}, {"Threads": "4"}],
        )
        self.assertEqual(
            [simulation.platform.name for simulation in self.simulations],
            ["CPU", "CPU"],
        )

    def test_cpu_without_explicit_threads_uses_platform_default(self):
        self.run_relax()
        self.assertEqual(self.simulations[0].properties, {})

    def test_other_platforms_do_not_receive_cpu_thread_property(self):
        for platform in ("CUDA", "OpenCL", "Reference"):
            with self.subTest(platform=platform):
                self.run_relax(platform=platform, threads=3)
                self.assertEqual(self.simulations[-1].properties, {})
                self.assertEqual(self.simulations[-1].platform.name, platform)

    def test_invalid_cpu_threads_fail_before_simulation_is_created(self):
        for threads in (0, -1, True, False, 1.5, "2"):
            with self.subTest(threads=threads):
                with self.assertRaisesRegex(ValueError, "positive integer"):
                    self.run_relax(threads=threads)
        self.app.Simulation.assert_not_called()


if __name__ == "__main__":
    unittest.main()
