"""Verify every bibliography entry against Crossref (DOI) or the arXiv API (eprint).

    python3 scientific_reports_paper/verify_refs.py scientific_reports_paper/refs.bib

Writes scientific_reports_paper/REFS_VERIFIED_2026_10_24.md: one row per key with the source
queried, the registered title and year, and whether the bib title matches (normalised token
overlap >= 0.85). Entries without a DOI or eprint are reported as unverifiable.
"""

from __future__ import annotations

import json
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

UA = {"User-Agent": "msp-refcheck/1.0"}


def entries(text: str) -> list[dict]:
    out = []
    for m in re.finditer(r"@(\w+)\{([^,]+),(.*?)\n\}", text, re.S):
        fields = dict(re.findall(r"(\w+)\s*=\s*\{(.*?)\},?\s*\n", m.group(3) + "\n", re.S))
        out.append({"type": m.group(1), "key": m.group(2).strip(), **fields})
    return out


def norm(s: str) -> list[str]:
    s = re.sub(r"\\[a-zA-Z]+|[{}\\'`^\"~]", "", s).lower()
    return re.findall(r"[a-z0-9]+", s)


def overlap(a: str, b: str) -> float:
    ta, tb = norm(a), set(norm(b))
    return sum(t in tb for t in ta) / max(1, len(ta))


def get(url: str) -> bytes:
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.read()


def crossref(doi: str) -> tuple[str, str]:
    j = json.loads(get("https://api.crossref.org/works/" + urllib.parse.quote(doi)))["message"]
    title = j.get("title", [""])[0]
    year = str((j.get("issued", {}).get("date-parts") or [[""]])[0][0])
    return title, year


def arxiv(eid: str) -> tuple[str, str]:
    x = get("http://export.arxiv.org/api/query?id_list=" + urllib.parse.quote(eid)).decode()
    ent = x.split("<entry>", 1)[1]
    title = re.sub(r"\s+", " ", re.search(r"<title>(.*?)</title>", ent, re.S).group(1)).strip()
    year = re.search(r"<published>(\d{4})", ent).group(1)
    return title, year


def main() -> None:
    bib = Path(sys.argv[1])
    rows = ["| key | source | registered title | year | match |", "|---|---|---|---|---|"]
    bad = 0
    for e in entries(bib.read_text()):
        try:
            if "doi" in e:
                src, (t, y) = f"doi:{e['doi']}", crossref(e["doi"])
            elif "eprint" in e:
                src, (t, y) = f"arXiv:{e['eprint']}", arxiv(e["eprint"])
                time.sleep(3)
            else:
                rows.append(f"| {e['key']} | none | - | - | UNVERIFIABLE |")
                bad += 1
                continue
            ok = overlap(e["title"], t) >= 0.85
            bad += not ok
            rows.append(f"| {e['key']} | {src} | {t} | {y} | {'ok' if ok else 'MISMATCH'} |")
        except Exception as ex:  # network or lookup failure is a verification failure
            rows.append(f"| {e['key']} | lookup failed | {type(ex).__name__}: {ex} | - | FAILED |")
            bad += 1
        print(rows[-1], flush=True)
    out = bib.parent / "REFS_VERIFIED_2026_10_24.md"
    out.write_text("# Reference verification (2026-10-24)\n\n"
                   f"Bibliography: `{bib.name}`. Problems: {bad}.\n\n" + "\n".join(rows) + "\n")
    print("problems:", bad)


if __name__ == "__main__":
    main()
