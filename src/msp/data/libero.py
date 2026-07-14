"""The LIBERO corpus: real grocery objects, RGB-D observations, rigid-body grasp outcomes.

This is the dataset the paper's experiments run on. Each scene is one scanned object, dropped on a
table and left to settle wherever gravity puts it, photographed from a ring of viewpoints, and
probed with candidate grasps that are rolled out in MuJoCo against the object's true geometry.

WHAT EACH SCENE CARRIES, AND WHY EVERY FIELD IS LOAD-BEARING

    observation   (V, 4, H, W)  RGB-D from V viewpoints. The camera sees the TRUE textured mesh.
    state         (14,)         the continuous state x. J(x) is a derivative with respect to it, so
                                a pipeline that never materializes x cannot compute contribution C2.
    object_index  ()            which grocery. Discrete -- deliberately NOT part of x, because
                                Theorem 4 is a statement about a tangent space and a categorical
                                label has none.
    actions       (Na, 7)       candidate grasps.
    succ/slip     (Na, 1)       the SIMULATOR's verdict, rolled out on the true convex decomposition.
    margin        (Na, 1)       Ferrari-Canny epsilon from the ANALYTIC tier, computed on a bounding
                                box -- i.e. on a RECONSTRUCTION, not on the truth.
    executable    (Na,)         whether the gripper can physically reach the pose at all.

The last three are what make the decisive experiment possible: `margin` is what the modular
grasp-planning literature would plan on, `succ` is what actually happens, and `executable` excludes
the grasps where both trivially agree because the hand cannot even get there. Asking how much the
first tells you about the second IS the paper (see `diagnostics/proxy_eval.py`).
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
from msp.oracle.libero_assets import LiberoObjectLibrary
from msp.oracle.libero_sim import LiberoGraspOracle

log = logging.getLogger(__name__)

__all__ = ["LiberoCorpusSpec", "LiberoGraspDataset", "generate_libero_corpus"]


#: Source files that DETERMINE THE OUTCOMES in a corpus: the grasp proposal, the analytic score,
#: the geometry, and the rollout. The cache key hashes their contents, so that fixing the physics
#: automatically invalidates every corpus generated under the old physics.
#:
#: This is not defensive tidiness. The cache key used to be a hash of the SPEC alone (n_scenes,
#: seed, image size...). A frame-convention bug in `libero_sim._rollout` was adding the object's
#: world position to the grasp point twice, so the simulator executed a grasp several centimetres
#: from the one that had been planned and scored; the nominal grasp succeeded 6% of the time. After
#: the fix it succeeds 99.6%. But the SPEC had not changed -- so every downstream run would have hit
#: the cache, silently reloaded the corpus generated under the broken physics, and reproduced the old
#: numbers perfectly. A fixed bug that keeps reporting its old results is worse than an open one.
_PHYSICS_SOURCES = (
    "oracle/libero_sim.py",  # the rollout: approach, close, lift, measure
    "oracle/libero_assets.py",  # the geometry the rollout collides
    "oracle/analytic.py",  # the grasp proposal AND the Ferrari-Canny score
)


def _physics_fingerprint() -> str:
    """Hash of the code that decides what a grasp DOES. Any edit invalidates the corpus cache."""
    root = Path(__file__).resolve().parent.parent  # .../src/msp
    h = hashlib.sha256()
    for rel in _PHYSICS_SOURCES:
        h.update((root / rel).read_bytes())
    return h.hexdigest()[:12]


@dataclass(frozen=True)
class LiberoCorpusSpec:
    n_scenes: int = 8000
    n_actions: int = 8
    image_size: int = 96
    n_views: int = 8  # |B|, the sensing action space. 1 = passive single camera.
    seed: int = 0
    objects: tuple[str, ...] = ()  # empty = all 13 HOPE groceries
    spread: float = 1.0  # proposal width. 1.0 = wide (A3/A8 need failures). See sample_actions.

    def key(self) -> str:
        raw = "|".join(f"{f}={getattr(self, f)}" for f in sorted(self.__dataclass_fields__))
        raw += f"|physics={_physics_fingerprint()}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]


def generate_libero_corpus(
    spec: LiberoCorpusSpec,
    cache_dir: Path,
    *,
    asset_root: str | Path | None = None,
    force: bool = False,
) -> Path:
    """Render the scenes, roll out the grasps, cache the result.

    Costs, so nobody is surprised: settling ~15 ms/scene, rendering ~2 ms/frame, and a grasp rollout
    ~8 ms. An 8k-scene, 8-view, 8-action corpus is therefore a few minutes and about 2 GB. It is
    generated ONCE and cached; regenerating it every epoch would dominate training by orders of
    magnitude.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"libero_{spec.key()}.pt"
    if path.exists() and not force:
        log.info("corpus cache hit: %s", path)
        return path

    lib = LiberoObjectLibrary(asset_root)
    sim = LiberoGraspOracle(library=lib, objects=list(spec.objects) or None)
    analytic = AnalyticGraspOracle(shape="box", noise=False)

    g = torch.Generator().manual_seed(spec.seed)

    log.info("settling %d scenes (%d objects)...", spec.n_scenes, sim.n_objects)
    oi, x = sim.sample_scenes(spec.n_scenes, generator=g)

    # The analytic tier plans on the bounding box -- the "reconstruction".
    analytic.set_base_half(sim.base_half(oi))
    actions = analytic.sample_actions(x, spec.n_actions, generator=g, spread=spec.spread)

    log.info("rendering %d scenes x %d view(s)...", spec.n_scenes, spec.n_views)
    views = [
        torch.cat(
            [
                sim.render(
                    oi[i : i + 128], x[i : i + 128],
                    height=spec.image_size, width=spec.image_size, view=v,
                )
                for i in range(0, spec.n_scenes, 128)
            ]
        )
        for v in range(spec.n_views)
    ]
    obs = torch.stack(views, dim=1)  # (N, V, 4, H, W)

    log.info("rolling out %d grasps on the TRUE meshes...", spec.n_scenes * spec.n_actions)
    y = sim.query_scenes(oi, x, actions)

    # The analytic prediction, on the RECONSTRUCTION. This is what the field would plan on.
    eps, _ = analytic._epsilon_quality(x, actions)  # noqa: SLF001 - the metric is the point
    executable = analytic._reachability(x, actions) > 0  # noqa: SLF001

    torch.save(
        {
            "observation": obs,
            "state": x,
            "object_index": oi,
            "object_names": sim.object_names(),
            "actions": actions,
            "succ": y.succ,  # simulator, TRUE geometry
            "slip": y.slip,
            "margin": eps.unsqueeze(-1),  # analytic, RECONSTRUCTED geometry
            "executable": executable,
            "spec": spec.__dict__,
            "provenance": sim.provenance(),
        },
        path,
    )
    log.info(
        "wrote %s | sim success %.3f | executable %.3f",
        path, float(y.succ.mean()), float(executable.float().mean()),
    )
    return path


class LiberoGraspDataset(Dataset[dict[str, Tensor]]):
    """A cached LIBERO corpus."""

    def __init__(self, path: str | Path, view: int = 0, max_train_views: int = 1) -> None:
        blob = torch.load(path, map_location="cpu", weights_only=False)
        self.all_obs: Tensor = blob["observation"]  # (N, V, 4, H, W)
        self.n_views: int = self.all_obs.shape[1]
        self.view = view
        self.max_train_views = min(max_train_views, self.n_views)
        self.obs: Tensor = self.all_obs[:, view]

        self.states: Tensor = blob["state"]
        self.object_index: Tensor = blob["object_index"]
        self.object_names: list[str] = blob["object_names"]
        self.actions: Tensor = blob["actions"]
        self.succ: Tensor = blob["succ"]
        self.slip: Tensor = blob["slip"]
        self.margin: Tensor = blob["margin"]
        self.executable: Tensor = blob["executable"]
        self.spec = blob["spec"]
        self.provenance = blob["provenance"]

    def all_views(self, idx: Tensor | slice) -> Tensor:
        """(n, V, 4, H, W) -- every viewpoint. Feeds Eq 17's rendered look-ahead."""
        return self.all_obs[idx]

    def __len__(self) -> int:
        return self.obs.shape[0]

    def __getitem__(self, i: int) -> dict[str, Tensor]:
        if self.max_train_views > 1:
            k = int(torch.randint(1, self.max_train_views + 1, (1,)))
            obs = self.all_obs[i, torch.randperm(self.n_views)[:k]]
        else:
            obs = self.obs[i]

        na = self.actions.shape[1]
        return {
            "observation": obs,
            "state": self.states[i],
            "actions": self.actions[i],
            "weights": torch.ones(na) / na,
            "succ": self.succ[i],
            "margin": self.margin[i],
            "slip": self.slip[i],
        }
