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
    #: How many of the ring's viewpoints to render per scene. 1 = the default single view
    #: (the passive experiment). >1 renders the whole sensing action space B, which is what
    #: Eq 17's rendered look-ahead needs -- IG_true is computed by actually LOOKING from each
    #: candidate viewpoint, which is only possible because x is known in simulation.
    n_views: int = 1

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

    log.info(
        "rendering %d scenes x %d view(s) at %dx%d...",
        spec.n_scenes, spec.n_views, spec.image_size, spec.image_size,
    )
    views = []
    for v in range(spec.n_views):
        views.append(
            torch.cat(
                [
                    sim.render(
                        states[i : i + 128],
                        height=spec.image_size,
                        width=spec.image_size,
                        view=v,
                    )
                    for i in range(0, spec.n_scenes, 128)
                ]
            )
        )
    obs = torch.stack(views, dim=1)  # (N, V, 4, H, W)

    log.info("querying M for %d grasps (simulator=%s)...", spec.n_scenes * spec.n_actions,
             spec.use_simulator)
    y = oracle.query(states, actions)

    torch.save(
        {
            "observation": obs,  # (N, V, 4, H, W)
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

    def __init__(
        self,
        path: str | Path,
        view: int = 0,
        max_train_views: int = 1,
    ) -> None:
        """
        Args:
            view: which viewpoint is the DEFAULT single observation.
            max_train_views: if > 1, each __getitem__ returns a RANDOM SUBSET of 1..max views.

                This is not data augmentation, it is what makes multi-view inference possible at
                all. Algorithm 2 acquires views ONE AT A TIME until the ambiguity U drops below
                tau_U, so the number of views the encoder will be handed at deployment is not known
                in advance. An encoder only ever trained on one view produces garbage when given
                two, and an encoder only ever trained on eight is useless on the first frame.
                Training on a random count teaches it to use whatever it gets -- and it is the
                mechanism by which a second view can reduce ambiguity at all.
        """
        blob = torch.load(path, map_location="cpu", weights_only=False)
        self.all_obs: Tensor = blob["observation"]  # (N, V, 4, H, W)
        self.n_views: int = self.all_obs.shape[1]
        self.view = view
        self.max_train_views = min(max_train_views, self.n_views)
        self.obs: Tensor = self.all_obs[:, view]  # (N, 4, H, W)
        self.states: Tensor = blob["state"]
        self.actions: Tensor = blob["actions"]
        self.succ: Tensor = blob["succ"]
        self.margin: Tensor = blob["margin"]
        self.slip: Tensor = blob["slip"]
        self.spec = blob["spec"]

    def all_views(self, idx: Tensor | slice) -> Tensor:
        """(n, V, 4, H, W) -- every viewpoint of the given scenes. Feeds Eq 17's look-ahead."""
        return self.all_obs[idx]

    def __len__(self) -> int:
        return self.obs.shape[0]

    def __getitem__(self, i: int) -> dict[str, Tensor]:
        if self.max_train_views > 1:
            k = int(torch.randint(1, self.max_train_views + 1, (1,)))
            picks = torch.randperm(self.n_views)[:k]
            obs = self.all_obs[i, picks]  # (k, 4, H, W)
        else:
            obs = self.obs[i]  # (4, H, W)

        return {
            "observation": obs,
            "state": self.states[i],
            "actions": self.actions[i],
            "weights": torch.ones(self.actions.shape[1]) / self.actions.shape[1],
            "succ": self.succ[i],
            "margin": self.margin[i],
            "slip": self.slip[i],
        }
