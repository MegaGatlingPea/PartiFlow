"""Isolated native metric execution for modified residues."""

from __future__ import annotations
import json
import sys
from pathlib import Path


def main():
    request, output = map(Path, sys.argv[1:])
    from .metrics import _compute_metrics
    from .runner import _write_json

    _write_json(output, _compute_metrics(**json.loads(request.read_text())))


if __name__ == "__main__":
    main()
