#!/usr/bin/env bash
# Clone third-party baselines at pinned commits. They are git-ignored: see baselines/README.md
# for why they are cloned rather than vendored (licence + reproducibility).
set -euo pipefail
cd "$(dirname "$0")"

clone_at() {
  local url=$1 dir=$2 commit=$3
  if [ -d "$dir/.git" ]; then
    echo "== $dir already present, skipping"
    return
  fi
  echo "== cloning $dir @ $commit"
  git clone "$url" "$dir"
  git -C "$dir" checkout --quiet "$commit"
}

# AnyDexGrasp (CC BY-NC 4.0). Code reference for the representation/decision split.
# NOTE: needs MinkowskiEngine v0.5 + torch 1.13 + CUDA 11.7 in its OWN conda env, and is unlikely
# to build on Blackwell. See baselines/README.md before investing time.
clone_at https://github.com/graspnet/AnyDexGrasp.git AnyDexGrasp c9c4a43

echo
echo "done. Baselines are NOT importable from the msp environment by design --"
echo "each needs its own env. See baselines/README.md."
