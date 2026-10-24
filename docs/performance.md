# Performance

## Sampling speed

![Sampling time by total tokens](figures/pepdbench_sampling_time_vs_tokens.png)

Sampling time per sample versus total tokens under the original benchmark settings.
The figure reports sampling cost; AF2 refolding and topology postprocessing are separate stages.

## Cyclic peptide design

![Cyclic peptide performance](figures/pepdbench_cyclic_peptide.png)

The original cyclic peptide benchmark compares projected successful designs over
24 hours (left) and structural diversity (right), using its original AF2 evaluation
and generation-budget definitions.

## Cyclic peptide docking

![CPSet50 docking performance](figures/cyclic-docking.png)

The paper's Fig. 7 summarizes retrospective docking with known peptide sequences
and topologies on CPSet50. Recovery uses the best candidate among 25 and requires
receptor-aligned peptide backbone RMSD < 2 angstroms. DockQ distributions use each
complex's minimum-RMSD candidate from that pool; white marks and dark segments
show medians and interquartile ranges.

These figures report the original study protocols, rather than a new benchmark
of this CLI package.
