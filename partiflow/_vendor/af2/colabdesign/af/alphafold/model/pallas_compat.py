"""Public Pallas Triton compiler options across JAX API generations.

JAX 0.4.29 accepts a nested ``triton`` dictionary in ``pallas_call``;
newer releases expose a compiler-parameter class in ``pallas.triton``.
Both forms configure the same Pallas Triton lowering and kernel options.
"""

try:
    import jax.experimental.pallas.triton as triton
except ModuleNotFoundError as error:
    if error.name != "jax.experimental.pallas.triton":
        raise
    triton = None


def triton_compiler_params(*, num_warps, num_stages):
    """Construct compiler options without changing the selected kernel."""
    if triton is None:
        return {"triton": {"num_warps": num_warps, "num_stages": num_stages}}
    cls = getattr(triton, "CompilerParams", None)
    if cls is None:
        cls = triton.TritonCompilerParams
    return cls(num_warps=num_warps, num_stages=num_stages)
