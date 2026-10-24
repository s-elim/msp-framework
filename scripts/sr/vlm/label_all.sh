#!/bin/bash
# Full ER-2 v2 labelling (study 3): val, train, test. Cached calls are free on rerun.
cd /nas/Selim/selim_sarowar/msp_framework
for s in val train test; do
  .venv/bin/python scripts/sr/vlm/gemini_label.py --split $s --model gemini-robotics-er-2-preview --prompt v2 --workers ${WORKERS:-24} --cap 60 || echo "FAILED $s"
done
echo ALLDONE
