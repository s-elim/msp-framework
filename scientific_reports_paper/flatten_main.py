"""Flatten main.tex into one standalone file for Overleaf and journal submission.

    python scientific_reports_paper/flatten_main.py OUT/main.tex

Inlines methods_body.tex and the two generated tables, and replaces every number macro of
numbers.tex (written by make_tables.py) with its value, so the result needs no other .tex file.
The split sources stay the editable originals; rerun this after any change to them.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEF = re.compile(r"^\\newcommand\{\\([A-Za-z]+)\}\{(.*)\}$")
INLINED = ("methods_body", "table_main", "table_ablation")


def macros() -> dict[str, str]:
    """Number macros of numbers.tex: name -> value (one-line, argument-free definitions)."""
    out = {}
    for line in (HERE / "numbers.tex").read_text().splitlines():
        m = DEF.match(line)
        if m:
            out[m.group(1)] = m.group(2)
        elif line.strip() and not line.startswith("%"):
            raise ValueError(f"numbers.tex: not a plain number macro: {line!r}")
    return out


def _body(name: str) -> str:
    """A .tex file without its leading comment lines (generator notes)."""
    return re.sub(r"\A(%.*\n)+", "", (HERE / f"{name}.tex").read_text()).rstrip("\n")


def flatten() -> str:
    tex = re.sub(r"\A(%.*\n)+", "", (HERE / "main.tex").read_text())
    tex = tex.replace("\\input{numbers}\n", "")
    for name in INLINED:
        tex = tex.replace(f"\\input{{{name}}}", _body(name))
    if "\\input{" in tex:
        raise ValueError("main.tex has an \\input this script does not inline")
    vals = macros()
    names = "|".join(sorted(vals, key=len, reverse=True))
    use = re.compile(rf"\\({names})(?![A-Za-z])(\\ )?")  # "\Macro\ word" -> "value word"
    tex = use.sub(lambda m: vals[m.group(1)] + (" " if m.group(2) else ""), tex)
    left = sorted(n for n in vals if re.search(rf"\\{n}(?![A-Za-z])", tex))
    if left:
        raise ValueError(f"number macros left after substitution: {left}")
    return tex


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: flatten_main.py OUT/main.tex")
    out = Path(sys.argv[1])
    out.write_text(flatten())
    print("wrote", out)


if __name__ == "__main__":
    main()
