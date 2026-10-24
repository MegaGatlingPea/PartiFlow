"""CPU staging checks for AF2 Pallas API compatibility; no model inference."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import textwrap
import unittest


ROOT = Path(__file__).resolve().parents[1]
HAS_JAX = importlib.util.find_spec("jax") is not None


@unittest.skipUnless(HAS_JAX, "requires the AF2 JAX runtime")
class AF2GPUCompatibilityTests(unittest.TestCase):
    def run_cpu(self, source):
        prefix = f"""
import sys
sys.path.insert(0, {str(ROOT / "partiflow/_vendor/af2")!r})
import jax
import jax.numpy as jnp
assert jax.default_backend() == "cpu"
"""
        result = subprocess.run(
            [sys.executable, "-c", prefix + textwrap.dedent(source)],
            cwd=ROOT,
            env={**os.environ, "JAX_PLATFORMS": "cpu", "CUDA_VISIBLE_DEVICES": ""},
            capture_output=True,
            text=True,
            timeout=60,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return result.stdout

    def test_legacy_and_current_compiler_option_forms(self):
        self.run_cpu("""
            from types import SimpleNamespace
            from unittest.mock import patch
            from colabdesign.af.alphafold.model import pallas_compat as compat

            options = {"num_warps": 4, "num_stages": 3}
            with patch.object(compat, "triton", None):
                assert compat.triton_compiler_params(**options) == {"triton": options}
                from jax.experimental import pallas as pl
                def copy_kernel(x, out):
                    out[...] = x[...]
                actual = pl.pallas_call(
                    copy_kernel, out_shape=jax.ShapeDtypeStruct((4,), jnp.float32),
                    compiler_params=compat.triton_compiler_params(**options),
                    interpret=True,
                )(jnp.arange(4, dtype=jnp.float32))
                assert bool(jnp.array_equal(actual, jnp.arange(4, dtype=jnp.float32)))

            class Params:
                def __init__(self, **kwargs):
                    self.options = kwargs
            for name in ("CompilerParams", "TritonCompilerParams"):
                with patch.object(compat, "triton", SimpleNamespace(**{name: Params})):
                    actual = compat.triton_compiler_params(**options)
                    assert isinstance(actual, Params)
                    assert actual.options == options
        """)

    def test_native_fast_kernels_stage_without_a_gpu(self):
        self.run_cpu("""
            from collections.abc import Mapping
            from colabdesign.af.alphafold.model.fast_attention import triatt_attn as attention
            from colabdesign.af.alphafold.model.fast_triangle import api as triangle

            def shape(size, dtype=jnp.bfloat16):
                return jax.ShapeDtypeStruct(size, dtype)

            def pallas_calls(value):
                if hasattr(value, "jaxpr"):
                    yield from pallas_calls(value.jaxpr)
                elif hasattr(value, "eqns"):
                    for equation in value.eqns:
                        if equation.primitive.name == "pallas_call":
                            yield equation.params
                        yield from pallas_calls(equation.params)
                elif isinstance(value, Mapping):
                    for item in value.values():
                        yield from pallas_calls(item)
                elif isinstance(value, (tuple, list)):
                    for item in value:
                        yield from pallas_calls(item)

            q, k = shape((1, 1, 17, 16)), shape((1, 1, 19, 16))
            bias, key_mask = shape((1, 17, 19)), shape((1, 19), jnp.bool_)
            def attend(q, k, v, bias, mask):
                return attention.attention(q, k, v, bias, mask, 1.0)
            forward = jax.make_jaxpr(attend)(q, k, k, bias, key_mask)
            calls = list(pallas_calls(forward))
            assert len(calls) == 2, calls
            assert forward.out_avals[0].shape == q.shape
            backward = jax.make_jaxpr(jax.grad(
                lambda q, k, v, b, m: attend(q, k, v, b, m).sum(),
                argnums=(0, 1, 2, 3),
            ))(q, k, k, bias, key_mask)
            assert len(list(pallas_calls(backward))) >= 6

            channels, intermediate = 32, 16
            params = {
                "ln_in_scale": shape((channels,), jnp.float32),
                "ln_in_offset": shape((channels,), jnp.float32),
                "w_proj": shape((channels, 2 * intermediate), jnp.float32),
                "w_gate": shape((channels, 2 * intermediate), jnp.float32),
                "b_proj": shape((2 * intermediate,), jnp.float32),
                "b_gate": shape((2 * intermediate,), jnp.float32),
                "ln_c_scale": shape((intermediate,), jnp.float32),
                "ln_c_offset": shape((intermediate,), jnp.float32),
                "w_out": shape((intermediate, channels), jnp.float32),
                "w_gl": shape((channels, channels), jnp.float32),
                "b_out": shape((channels,), jnp.float32),
                "b_gl": shape((channels,), jnp.float32),
            }
            for equation in ("ikc,jkc->ijc", "kjc,kic->ijc"):
                traced = jax.make_jaxpr(lambda a, m, p: triangle._pallas_triangle(
                    a, m, p, equation=equation,
                ))(shape((17, 17, channels)), shape((17, 17)), params)
                triangle_calls = list(pallas_calls(traced))
                assert len(triangle_calls) == 2, triangle_calls
                assert traced.out_avals[0].shape == (17, 17, channels)
                calls.extend(triangle_calls)
            for call in calls:
                options = call["compiler_params"]
                if isinstance(options, dict):
                    assert options["triton"]["num_warps"] == 4
                    assert options["triton"]["num_stages"] in (2, 3)
            print("AF2 attention forward/backward and both triangle directions staged as Pallas calls")
        """)


if __name__ == "__main__":
    unittest.main()
