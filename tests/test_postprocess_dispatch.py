"""CPU-only scheduler contracts; these do not initialize a native SDK."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from partiflow.postprocess import dispatch as module
from partiflow.postprocess import runner


@dataclass
class Item:
    name: str
    sdk_profile: str = "standard_beta"
    needs_gpu: bool = False


class Progress:
    instances = []

    def __init__(self, **kwargs):
        self.total = kwargs["total"]
        self.n = 0
        self.closed = False
        self.instances.append(self)

    def update(self, amount):
        self.n += amount

    def close(self):
        self.closed = True


class Ref:
    def __init__(self, actor, method, when, item=None):
        self.actor, self.method, self.when, self.item = actor, method, when, item


class Actor:
    pass


class FakeRay:
    def __init__(self, *, initialized=False, cpus=256, gpus=0, delays=None,
                 actions=None, startup=0, kill_delay=0, bad_start=None):
        self.initialized = initialized
        self.cpus, self.gpus = cpus, gpus
        self.delays, self.actions = delays or {}, actions or {}
        self.startup, self.kill_delay, self.bad_start = startup, kill_delay, bad_start
        self.now = 0.0
        self.actors, self.references, self.calls, self.recovered = [], [], [], []
        self.options, self.killed, self.shutdowns = [], [], 0
        self.init_kwargs = None
        self.published = {}
        self.advance_first_ready = 0
        self.peak_live = self.peak_gpu = 0

    def is_initialized(self):
        return self.initialized

    def init(self, **kwargs):
        self.init_kwargs = kwargs

    def cluster_resources(self):
        return {"CPU": self.cpus, "GPU": self.gpus}

    available_resources = cluster_resources

    def shutdown(self):
        self.shutdowns += 1

    def remote(self, **options):
        self.options.append(options)

        def decorator(worker_class):
            def create(settings):
                actor = Actor()
                actor.index = len(self.actors)
                actor.settings = settings
                actor.killed = False
                actor.processed = 0
                self.actors.append(actor)
                live = [a for a in self.actors if not a.killed]
                self.peak_live = max(self.peak_live, len(live))
                self.peak_gpu = max(self.peak_gpu, sum(a.settings["needs_gpu"] for a in live))

                def reference(method, item=None):
                    delay = self.startup if method == "ready" else 0
                    if method == "process":
                        assert item.sdk_profile == settings["sdk_profile"]
                        assert item.needs_gpu == settings["needs_gpu"]
                        self.calls.append((actor.index, item.name, self.now))
                        delay = self.delays.get(item.name, 1)
                        if self.actions.get(item.name, "").startswith("timeout"):
                            delay = 1000
                    ref = Ref(actor, method, self.now + delay, item)
                    self.references.append(ref)
                    return ref

                actor.ready = SimpleNamespace(remote=lambda: reference("ready"))
                actor.stats = SimpleNamespace(remote=lambda: reference("stats"))
                actor.process = SimpleNamespace(remote=lambda item: reference("process", item))
                return actor

            return SimpleNamespace(remote=create)

        return decorator

    def wait(self, refs, *, num_returns, timeout):
        assert refs
        ordered = sorted(refs, key=lambda ref: ref.when)
        target = ordered[min(num_returns, len(ordered)) - 1].when
        cutoff = float("inf") if timeout is None else self.now + timeout
        self.now = max(self.now, min(target, cutoff))
        finished = [ref for ref in ordered if ref.when <= self.now][:num_returns]
        return finished, [ref for ref in refs if ref not in finished]

    def get(self, ref):
        actor = ref.actor
        if actor.killed:
            raise RuntimeError("actor terminated")
        if ref.method in {"ready", "stats"}:
            if ref.method == "ready" and self.bad_start == "exception":
                raise ValueError("SDK startup failed")
            if ref.method == "ready" and self.advance_first_ready:
                self.now += self.advance_first_ready
                self.advance_first_ready = 0
            info = dict(pid=1000 + actor.index, sdk_profile=actor.settings["sdk_profile"],
                        initializations=0 if actor.settings["sdk_profile"] == "native_ref2015" else 1,
                        processed=actor.processed)
            if self.bad_start == "profile":
                info["sdk_profile"] = "wrong_profile"
            if self.bad_start == "pid":
                info["pid"] = 0
            return info
        item = ref.item
        actor.processed += 1
        action = self.actions.get(item.name)
        if action == "published_crash":
            self.published[item.name] = {"status": "success", "name": item.name}
        if action in {"crash", "published_crash"}:
            raise RuntimeError("native process died")
        if action == "filesystem":
            raise OSError("destination is unavailable")
        if action == "wrapped_filesystem":
            error = RuntimeError("remote filesystem error")
            error.cause = PermissionError("destination is unavailable")
            raise error
        if action == "invalid":
            return None
        return {"name": item.name, "status": "failed" if action == "failed" else "success"}

    def kill(self, actor, *, no_restart):
        assert no_restart
        actor.killed = True
        self.killed.append(actor.index)
        for ref in self.references:
            if ref.actor is actor and ref.when > self.now:
                if ref.item is not None and self.actions.get(ref.item.name) == "timeout_published":
                    self.published[ref.item.name] = {"status": "success", "name": ref.item.name}
                ref.when = self.now + self.kill_delay

    def recover(self, item, settings, reason):
        assert settings["sdk_profile"] == item.sdk_profile
        assert settings["needs_gpu"] == item.needs_gpu
        related = [ref for ref in self.references if ref.item is item]
        assert related and all(ref.actor.killed and ref.when <= self.now for ref in related)
        self.recovered.append((item.name, reason))
        return {**self.published.get(item.name, {"status": "failed", "name": item.name}),
                "metrics_status": "unavailable", "error": reason}


class DispatchTests(unittest.TestCase):
    def setUp(self):
        Progress.instances = []
        self.settings = dict(workers=2, threads=1, timeout=10, seed=2026, platform="CPU")

    def execute(self, items, *, ray=None, callback=None, settings=None):
        ray = ray or FakeRay()
        with (patch.object(module, "cpu_budget", return_value=256),
              patch.object(module.time, "monotonic", side_effect=lambda: ray.now),
              patch.object(runner, "PostprocessWorker", object, create=True),
              patch.object(runner, "recover_actor_failure", side_effect=ray.recover, create=True),
              patch("tqdm.tqdm", Progress)):
            result = module.dispatch_items(items, settings or self.settings,
                                           on_result=callback, ray_module=ray)
        return result, ray

    def test_empty_does_not_load_or_start_runtime(self):
        self.assertEqual(module.dispatch_items([], {}), ([], []))

    def test_profile_metadata_defaults_and_alias(self):
        self.assertEqual(module._profile(SimpleNamespace()), ("standard_beta", False))
        self.assertEqual(module._profile(SimpleNamespace(runtime_profile="native_ref2015")),
                         ("native_ref2015", False))
        with self.assertRaises(ValueError):
            module._profile(SimpleNamespace(sdk_profile=""))

    def test_timeout_configuration_is_validated_before_runtime_start(self):
        for value in (0, -1, True, float("inf"), float("nan"), "10"):
            with self.subTest(timeout=value), self.assertRaises(ValueError):
                self.execute([Item("a")], settings={**self.settings, "timeout": value})

    def test_resource_validation(self):
        for options in ({"workers": 0}, {"workers": True}, {"threads": 0}, {"threads": True}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                module.worker_count({**self.settings, **options}, 2, 4)
        with self.assertRaisesRegex(ValueError, "CPU budget"):
            module.worker_count(dict(threads=5, workers=None), 2, 4)

    def test_affinity_and_job_allocation_cap(self):
        values = {"SLURM_CPUS_PER_TASK": "12", "SLURM_CPUS_ON_NODE": "16",
                  "SLURM_JOB_CPUS_PER_NODE": "10(x2),10"}
        with (patch.object(module.os, "cpu_count", return_value=32),
              patch.object(module.os, "sched_getaffinity", return_value=set(range(24))),
              patch.dict(module.os.environ, values)):
            self.assertEqual(module.cpu_budget(), 10)

    def test_dynamic_refill_and_original_result_order(self):
        ray = FakeRay(delays={"a": 9, "b": 1, "c": 1})
        events = []
        (results, audit), ray = self.execute([Item(n) for n in "abc"], ray=ray,
            callback=lambda index, result: events.append((index, result["name"])))
        self.assertEqual([r["name"] for r in results], list("abc"))
        self.assertEqual([index for index, _ in events], [1, 2, 0])
        self.assertLess(next(t for _, name, t in ray.calls if name == "c"), 9)
        self.assertEqual(sorted(a["processed"] for a in audit), [1, 2])
        self.assertTrue(all(a["initializations"] == 1 for a in audit))
        self.assertEqual(ray.shutdowns, 1)
        self.assertTrue(all(a.killed for a in ray.actors))
        self.assertEqual(Progress.instances[-1].n, 3)
        self.assertTrue(Progress.instances[-1].closed)

    def test_local_runtime_cpu_reservation_uses_active_pool(self):
        for workers, threads, jobs, cpus in ((3, 2, 20, 6), (None, 2, 2, 4), (20, 2, 1, 2)):
            with self.subTest(workers=workers, jobs=jobs):
                (_, _), ray = self.execute([Item(str(i)) for i in range(jobs)],
                    settings={**self.settings, "workers": workers, "threads": threads})
                self.assertEqual(ray.init_kwargs["num_cpus"], cpus)
                self.assertEqual(ray.init_kwargs["num_gpus"], 0)
                self.assertEqual(ray.init_kwargs["address"], "local")
                self.assertEqual(ray.options[0]["runtime_env"]["env_vars"]["OMP_NUM_THREADS"], str(threads))

    def test_preinitialized_runtime_caps_available_cpu_and_is_not_closed(self):
        ray = FakeRay(initialized=True, cpus=3)
        (_, audit), ray = self.execute([Item(str(i)) for i in range(5)], ray=ray,
                                      settings={**self.settings, "workers": 20, "threads": 2})
        self.assertIsNone(ray.init_kwargs)
        self.assertEqual(len(audit), 1)
        self.assertEqual(ray.shutdowns, 0)
        self.assertTrue(all(a.killed for a in ray.actors))

    def test_profiles_are_immutable_and_idle_slot_is_replaced(self):
        items = [Item("a"), Item("b", "native_ref2015"), Item("c")]
        (results, audit), ray = self.execute(items, settings={**self.settings, "workers": 1})
        self.assertEqual([r["name"] for r in results], ["a", "b", "c"])
        self.assertEqual([name for _, name, _ in ray.calls], ["a", "c", "b"])
        self.assertEqual([a["sdk_profile"] for a in audit], ["standard_beta", "native_ref2015"])
        self.assertEqual([a["initializations"] for a in audit], [1, 0])
        self.assertEqual(ray.peak_live, 1)

    def test_compatible_profiles_can_run_concurrently(self):
        items = [Item("a"), Item("b", "native_ref2015"), Item("c"), Item("d", "native_ref2015")]
        (results, audit), ray = self.execute(items)
        self.assertEqual(len(results), 4)
        self.assertEqual(len(audit), 2)
        self.assertEqual(Counter(a["sdk_profile"] for a in audit), {"standard_beta": 1, "native_ref2015": 1})
        self.assertEqual(ray.peak_live, 2)

    def test_gpu_fraction_and_mixed_resource_cap(self):
        ray = FakeRay(gpus=2 * module.GPU_FRACTION)
        items = [Item("cpu" + str(i)) for i in range(3)]
        items += [Item("gpu" + str(i), "native_ref2015", True) for i in range(5)]
        (results, _), ray = self.execute(items, ray=ray,
            settings={**self.settings, "workers": 6, "platform": "CUDA"})
        self.assertEqual(len(results), 8)
        self.assertLessEqual(ray.peak_live, 5)
        self.assertEqual(ray.peak_gpu, 2)
        self.assertNotIn("num_gpus", ray.init_kwargs)
        self.assertEqual({o["num_gpus"] for o in ray.options}, {0, module.GPU_FRACTION})

    def test_missing_gpu_fails_before_any_sample_starts(self):
        ray = FakeRay()
        with self.assertRaisesRegex(ValueError, "GPU resources"):
            self.execute([Item("gpu", "native_ref2015", True)], ray=ray)
        self.assertEqual(ray.calls, [])
        self.assertEqual(ray.shutdowns, 1)
        self.assertTrue(Progress.instances[-1].closed)

    def test_crash_is_local_and_next_sample_uses_replacement(self):
        ray = FakeRay(actions={"a": "crash"})
        (results, _), ray = self.execute([Item("a"), Item("b")], ray=ray,
                                       settings={**self.settings, "workers": 1})
        self.assertEqual([r["status"] for r in results], ["failed", "success"])
        self.assertEqual([name for _, name, _ in ray.calls], ["a", "b"])
        self.assertEqual(len(ray.actors), 2)
        self.assertEqual(len(ray.recovered), 1)

    def test_published_geometry_survives_actor_failure(self):
        for action in ("published_crash", "timeout_published"):
            with self.subTest(action=action):
                ray = FakeRay(actions={"a": action}, kill_delay=.5)
                (results, _), ray = self.execute([Item("a"), Item("b")], ray=ray,
                    settings={**self.settings, "workers": 1, "timeout": 2})
                self.assertEqual([r["status"] for r in results], ["success", "success"])
                self.assertEqual(results[0]["metrics_status"], "unavailable")
                self.assertEqual([name for _, name, _ in ray.calls], ["a", "b"])

    def test_timeout_waits_for_termination_before_recovery(self):
        ray = FakeRay(actions={"a": "timeout"}, kill_delay=.5)
        (results, _), ray = self.execute([Item("a"), Item("b")], ray=ray,
            settings={**self.settings, "workers": 1, "timeout": 2})
        self.assertEqual(results[0]["status"], "failed")
        self.assertIn("timeout", ray.recovered[0][1])
        self.assertGreaterEqual(next(t for _, name, t in ray.calls if name == "b"), 2.5)

    def test_startup_failure_and_invalid_information_are_run_fatal(self):
        for failure in ("exception", "profile", "pid"):
            with self.subTest(failure=failure):
                ray = FakeRay(bad_start=failure)
                with self.assertRaisesRegex(RuntimeError, "startup"):
                    self.execute([Item("a"), Item("b")], ray=ray)
                self.assertEqual(ray.calls, [])
                self.assertTrue(all(a.killed for a in ray.actors))
                self.assertTrue(Progress.instances[-1].closed)

    def test_startup_timeout_stops_all_actors(self):
        ray = FakeRay(startup=100, kill_delay=.5)
        with self.assertRaisesRegex(RuntimeError, "startup timeout"):
            self.execute([Item("a"), Item("b")], ray=ray,
                         settings={**self.settings, "timeout": 2})
        self.assertTrue(all(a.killed for a in ray.actors))
        self.assertTrue(all(ref.when <= ray.now for ref in ray.references))

    def test_already_ready_startup_is_not_misreported_as_timeout(self):
        ray = FakeRay(startup=1)
        ray.advance_first_ready = 1.5
        (results, _), _ = self.execute([Item("a"), Item("b")], ray=ray,
            settings={**self.settings, "timeout": 2})
        self.assertEqual([r["status"] for r in results], ["success", "success"])
        self.assertFalse(ray.recovered)

    def test_already_completed_process_is_not_misreported_as_timeout(self):
        ray = FakeRay()
        def callback(index, result):
            if index == 0:
                ray.now += 1.5
        (results, _), _ = self.execute([Item("a"), Item("b")], ray=ray, callback=callback,
            settings={**self.settings, "timeout": 2})
        self.assertEqual([r["status"] for r in results], ["success", "success"])
        self.assertFalse(ray.recovered)

    def test_filesystem_errors_are_not_scientific_failures(self):
        for action in ("filesystem", "wrapped_filesystem"):
            with self.subTest(action=action):
                ray = FakeRay(actions={"a": action}, delays={"b": 20}, kill_delay=.5)
                with self.assertRaisesRegex(OSError, "destination"):
                    self.execute([Item("a"), Item("b")], ray=ray)
                self.assertFalse(ray.recovered)
                self.assertTrue(all(a.killed for a in ray.actors))
                self.assertTrue(all(ref.when <= ray.now for ref in ray.references))

    def test_callback_error_waits_for_all_inflight_calls_before_return(self):
        ray = FakeRay(delays={"a": 1, "b": 20}, kill_delay=.5)
        def callback(index, result):
            raise OSError("CSV publication failed")
        with self.assertRaisesRegex(OSError, "CSV publication"):
            self.execute([Item("a"), Item("b"), Item("c")], ray=ray, callback=callback)
        self.assertEqual([name for _, name, _ in ray.calls], ["a", "b"])
        self.assertTrue(all(a.killed for a in ray.actors))
        self.assertTrue(all(ref.when <= ray.now for ref in ray.references))
        self.assertTrue(Progress.instances[-1].closed)

    def test_sample_failure_does_not_replace_healthy_worker(self):
        ray = FakeRay(actions={"a": "failed"})
        (results, audit), ray = self.execute([Item("a"), Item("b")], ray=ray,
            settings={**self.settings, "workers": 1})
        self.assertEqual([r["status"] for r in results], ["failed", "success"])
        self.assertEqual(len(audit), 1)
        self.assertFalse(ray.recovered)

    def test_invalid_result_replaces_worker(self):
        ray = FakeRay(actions={"a": "invalid"})
        (results, _), ray = self.execute([Item("a"), Item("b")], ray=ray,
            settings={**self.settings, "workers": 1})
        self.assertEqual([r["status"] for r in results], ["failed", "success"])
        self.assertEqual(len(ray.actors), 2)


if __name__ == "__main__":
    unittest.main()
