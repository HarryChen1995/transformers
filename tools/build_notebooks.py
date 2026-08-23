"""Build .ipynb notebooks from percent-format Python sources.

Sources live in notebooks/_src/*.py and use the jupytext "percent" format:

    # %% [markdown]
    # # A heading
    # some prose

    # %%
    print("a code cell")

Run:
    python3 tools/build_notebooks.py            # build all
    python3 tools/build_notebooks.py 03 07      # build only matching sources
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT / "notebooks" / "_src"
OUT_DIR = ROOT / "notebooks"


def parse_cells(text: str) -> list[tuple[str, str]]:
    """Split percent-format source into (kind, source) cells."""
    cells: list[tuple[str, list[str]]] = []
    kind = "code"
    buf: list[str] = []

    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("# %%"):
            if buf:
                cells.append((kind, buf))
            kind = "markdown" if "[markdown]" in stripped else "code"
            buf = []
            continue
        buf.append(line)
    if buf:
        cells.append((kind, buf))

    out: list[tuple[str, str]] = []
    for kind, lines in cells:
        if kind == "markdown":
            # strip the leading "# " comment marker from each prose line
            body = []
            for ln in lines:
                if ln.startswith("# "):
                    body.append(ln[2:])
                elif ln.strip() == "#":
                    body.append("")
                else:
                    body.append(ln)
            source = "\n".join(body).strip("\n")
        else:
            source = "\n".join(lines).strip("\n")
        if source.strip():
            out.append((kind, source))
    return out


def to_notebook(cells: list[tuple[str, str]]) -> dict:
    nb_cells = []
    for kind, source in cells:
        lines = source.splitlines(keepends=True)
        cell = {"cell_type": kind, "metadata": {}, "source": lines}
        if kind == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
        nb_cells.append(cell)
    return {
        "cells": nb_cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main(argv: list[str]) -> int:
    sources = sorted(SRC_DIR.glob("*.py"))
    if argv:
        sources = [p for p in sources if any(a in p.name for a in argv)]
    if not sources:
        print("no matching sources in", SRC_DIR)
        return 1
    for src in sources:
        nb = to_notebook(parse_cells(src.read_text()))
        out = OUT_DIR / (src.stem + ".ipynb")
        out.write_text(json.dumps(nb, indent=1) + "\n")
        n_md = sum(c["cell_type"] == "markdown" for c in nb["cells"])
        n_code = sum(c["cell_type"] == "code" for c in nb["cells"])
        print(f"{out.relative_to(ROOT)}  ({n_md} md, {n_code} code)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
