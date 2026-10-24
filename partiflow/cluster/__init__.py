"""CPU structural clustering of selected current design backbones."""

from __future__ import annotations


def run(args):
    """Run the public ``partiflow cluster`` argument namespace."""
    from .runner import run_paths

    return run_paths(
        args.paths,
        tasks=getattr(args, "task", None) or (),
        criteria=getattr(args, "criteria", None) or (),
        threshold=getattr(args, "tm_threshold", 0.5),
        workers=getattr(args, "workers", 16),
        executable=getattr(args, "usalign", None),
        plan=getattr(args, "plan", False),
    )
