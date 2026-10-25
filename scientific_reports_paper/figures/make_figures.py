"""Regenerate every figure of the Scientific Reports manuscript from the results files.

    cd /nas/Selim/selim_sarowar/msp_framework
    source scripts/sr/env.sh && $PY scientific_reports_paper/figures/make_figures.py [--only fig2]

Outputs beside this script, each as vector PDF (fonts embedded) and 300 dpi PNG:
    fig1_teaser        whole LM-O frame, point-estimate vs posterior planning (fig_lmo.py)
    fig2_method        the pipeline on one LM-Grasp test scene (fig_method.py)
    fig3_pose_gap      pose error vs grasp success (fig_plots.py)
    fig4_results       matched comparison, selective execution, certificates (fig_plots.py)
    fig5_analysis      posterior calibration, second view (fig_plots.py)
    fig6_qualitative   whole LM-O frames (fig_lmo.py)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import fig_lmo
import fig_method
import fig_plots


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="", help="comma-separated: fig1,fig2,...,fig6")
    a = ap.parse_args()
    want = set(a.only.split(",")) if a.only else {f"fig{k}" for k in range(1, 7)}
    if want & {"fig1", "fig6"}:
        top, rest = fig_lmo.choose_frames()
        print("LM-O teaser frame", top, "qualitative frames", rest)
        if "fig1" in want:
            fig_lmo.teaser(top)
        if "fig6" in want:
            fig_lmo.qualitative(rest)
    if "fig2" in want:
        fig_method.draw()
    if "fig3" in want:
        fig_plots.fig3_pose_gap()
    if "fig4" in want:
        fig_plots.fig4_results()
    if "fig5" in want:
        fig_plots.fig5_analysis()


if __name__ == "__main__":
    main()
