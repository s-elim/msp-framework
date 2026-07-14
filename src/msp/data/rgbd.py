"""RGB-D grasp dataset: the observation is an IMAGE, rendered from the physics scene.

This is the module that turns MSP from a framework into an experiment. Before it, the "observation"
the encoder consumed was a 32-dimensional random linear projection of the state -- a stand-in that
let the pipeline be tested but could not test the pipeline's central claim.

The claim is this: a camera cannot resolve the world state, so perception must output a calibrated
BELIEF and not a pose. That is only falsifiable against a real image, where the things that decide
grasp success -- friction, mass, the centre of mass -- are genuinely invisible.

Generation is done ONCE, up front, and cached to disk: rendering is ~80 ms/frame and a rigid-body
rollout is ~10 ms/grasp, so a 20k-scene corpus is minutes of work that must not be repeated every
epoch.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor
from torch.utils.data import Dataset

from msp.oracle.analytic import AnalyticGraspOracle
from msp.oracle.composite import CompositeOracle
from msp.oracle.mujoco_sim import MuJoCoOracle

log = logging.getLogger(__name__)

__all__ = ["RGBDGraspDataset", "generate_corpus"]


@dataclass(frozen=True)
class CorpusSpec:
    """Everything that determines the contents of a corpus. Two specs that hash the same MUST
    produce the same data, which is what makes the cache safe to trust."""

    n_scenes: int
    n_actions: int
    shape: str
    image_size: int
    seed: int
    use_simulator: bool

    def key(self) -> str:
        raw = "|".join(str(getattr(self, f)) for f in sorted(self.__dataclass_fields__))
        return hashlib.sha256(raw.encode()).hexdigest()[:16]


def generate_corpus(
    spec: CorpusSpec,
    cache_dir: Path,
    *,
    force: bool = False,
) -> Path:
    """Render the scenes, query M, and cache the result. Returns the cache path.

    `use_simulator=True` takes success and slip from MuJoCo rollouts (the honest labels, ~10 ms
    per grasp). `False` takes them from the analytic tier alone (instant, but its false-positive
    rate against the simulator is ~0.62 -- see CompositeOracle.tier_gap). Use False for smoke tests
    and True for anything that goes in the paper.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"corpus_{spec.key()}.pt"
    if path.exists() and not force:
        log.info("corpus cache hit: %s", path)
        return path

    analytic = AnalyticGraspOracle(shape=spec.shape)
    sim = MuJoCoOracle(shape=spec.shape)
    oracle = CompositeOracle(analytic, sim if spec.use_simulator else None)

    g = torch.Generator().manual_seed(spec.seed)
    states = analytic.sample_states(spec.n_scenes, generator=g)
    actions = analytic.sample_actions(states, spec.n_actions, generator=g)

    log.info("rendering %d RGB-D scenes at %dx%d...", spec.n_scenes, spec.image_size, spec.image_size)
    obs = torch.cat(
        [
            sim.render(states[i : i + 64], height=spec.image_size, width=spec.image_size)
            for i in range(0, spec.n_scenes, 64)
        ]
    )

    log.info("querying M for %d grasps (simulator=%s)...", spec.n_scenes * spec.n_actions,
             spec.use_simulator)
    y = oracle.query(states, actions)

    torch.save(
        {
            "observation": obs,  # (N, 4, H, W)
            "state": states,  # (N, d_X)   -- J(x) needs it
            "actions": actions,  # (N, Na, 7)
            "succ": y.succ,
            "margin": y.margin,
            "slip": y.slip,
            "spec": spec.__dict__,
        },
        path,
    )
    log.info("wrote %s  (success rate %.3f)", path, float(y.succ.mean()))
    return path


class RGBDGraspDataset(Dataset[dict[str, Tensor]]):
    """A cached RGB-D corpus. Loads into RAM: at 128x128, 20k scenes is ~4 GB."""

    def __init__(self, path: str | Path) -> None:
        blob = torch.load(path, map_location="cpu", weights_only=False)
        self.obs: Tensor = blob["observation"]
        self.states: Tensor = blob["state"]
        self.actions: Tensor = blob["actions"]
        self.succ: Tensor = blob["succ"]
        self.margin: Tensor = blob["margin"]
        self.slip: Tensor = blob["slip"]
        self.spec = blob["spec"]

    def __len__(self) -> int:
        return self.obs.shape[0]

    def __getitem__(self, i: int) -> dict[str, Tensor]:
        return {
            "observation": self.obs[i],
            "state": self.states[i],
            "actions": self.actions[i],
            "weights": torch.ones(self.actions.shape[1]) / self.actions.shape[1],
            "succ": self.succ[i],
            "margin": self.margin[i],
            "slip": self.slip[i],
        }
