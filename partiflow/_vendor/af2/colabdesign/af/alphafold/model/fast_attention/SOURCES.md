Sources: Anthropic uplifting-biomolecular-modeling, commit
f4f62fa6592ae4938d49b1757bea0cfeff9f468e,
common/opt_core/opt_core/kernels/pallas_triatt/{triatt_attn,attbwd_dkdv}.py.
Kernel arithmetic is copied unchanged. Pallas BlockSpec calls use keyword
arguments, and the local pallas_compat helper supplies the compiler-option
representation accepted by the installed JAX API (including JAX 0.4.29).
The local modules.Attention adapter supplies projected tensors.
