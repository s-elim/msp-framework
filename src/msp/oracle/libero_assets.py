"""Load LIBERO / robosuite object assets into a standalone MuJoCo scene.

WHY THIS MATTERS TO THE SCIENCE, and not just to the engineering.

Until now the objects were boxes and cylinders. On a box, "plan a grasp on the reconstructed
geometry" and "plan a grasp on the true geometry" are the same thing, so the paper's central
indictment --

    "Force closure computed on the estimated geometry is a valid success criterion" is WRONG:
    it is an analytic proxy evaluated on a hallucinated surface. A self-consistent but wrong
    reconstruction passes the check and fails the lift.   (blueprint, wrong-assumption #11)

-- cannot even be *stated*, let alone measured. There is no gap between the true shape and its
convex approximation when the true shape IS a box.

LIBERO's objects are real scanned groceries and household items. Their MJCF ships two geometries:

    group="1"   a textured triangle MESH -- what the camera sees, and what the simulator
                *would* use if we asked it to (we do not: mesh-mesh contact is expensive and
                LIBERO itself does not use it for collision either)
    group="0"   a CONVEX DECOMPOSITION into ~20 boxes -- what the simulator actually collides

That split is exactly the experiment. The analytic Ferrari-Canny tier plans on an ORIENTED
BOUNDING BOX fitted to the object -- a deliberately crude "reconstruction", of the kind a pose-and-
shape pipeline produces. The simulator rolls the grasp out against the true convex decomposition.
The disagreement between them is not an artefact to be minimized; it IS the measurement the paper
is about, and `CompositeOracle.tier_gap()` reports it.

MERGING, NOT INCLUDING. A LIBERO object XML is a complete <mujoco> document with its own <asset>
block, so `<include>`-ing it inside a <worldbody> fails with `Element 'asset', line 0`. The asset
declarations must be hoisted into the host scene's <asset> and the body grafted into its
<worldbody>, with every mesh/texture path made absolute. That is what robosuite does internally and
what this module does here.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

__all__ = ["LiberoObject", "LiberoObjectLibrary", "DEFAULT_ASSET_ROOTS"]


#: Where to look for LIBERO assets, in order of preference.
#:
#: The VENDORED copy under `assets/libero/` is first and is what the experiments use. It contains
#: only the 13 HOPE grocery objects, with the unused source art stripped and the textures
#: downsampled to 512px (we render at 96x96, so a 2048px texture carries 400x more resolution than
#: a pixel can hold). 30 MB instead of 100 MB, and the repository is self-contained: nobody needs a
#: LIBERO checkout to reproduce a number.
#:
#: The upstream LIBERO installs are fallbacks and are ONLY EVER READ. Nothing in this codebase
#: writes to them.
DEFAULT_ASSET_ROOTS: tuple[str, ...] = (
    str(Path(__file__).resolve().parents[3] / "assets" / "libero"),  # vendored, self-contained
    "/data/selim_sarowar/LIBERO-X/libero/libero/assets",
    "/data/selim_sarowar/GST-VLA/LIBERO/libero/libero/assets",
)

#: Categories that contain free-standing, table-top, GRASPABLE objects. Deliberately excludes
#: `articulated_objects` (microwaves, cabinets -- they are fixtures, not things you pick up) and
#: `scenes`.
GRASPABLE_CATEGORIES: tuple[str, ...] = (
    "stable_hope_objects",  # HOPE: scanned grocery items, the YCB-adjacent set
    "stable_scanned_objects",  # Google scanned objects
    "turbosquid_objects",
)


@dataclass(frozen=True)
class LiberoObject:
    """One graspable asset, with the geometry both tiers of M need."""

    name: str
    category: str
    xml_path: Path

    #: Convex decomposition used for CONTACT: (n, 3) half-extents, (n, 3) positions,
    #: (n, 4) quaternions [w,x,y,z], all in the object frame.
    box_sizes: np.ndarray
    box_pos: np.ndarray
    box_quat: np.ndarray

    friction: float
    density: float

    @property
    def n_boxes(self) -> int:
        return int(self.box_sizes.shape[0])

    def aabb_half_extents(self) -> np.ndarray:
        """Axis-aligned bounding half-extents of the convex decomposition. (3,).

        This is the "reconstruction" the ANALYTIC tier plans on. It is deliberately crude: a real
        pose-and-shape pipeline outputs something of about this fidelity, and the whole point is to
        measure what that crudeness costs when the grasp is actually executed.
        """
        lo = (self.box_pos - self.box_sizes).min(axis=0)
        hi = (self.box_pos + self.box_sizes).max(axis=0)
        return (hi - lo) / 2.0

    def is_graspable(self, jaw_half_width: float = 0.044) -> bool:
        """Can a parallel jaw straddle it at all? Objects wider than the gripper are not hard
        grasps, they are impossible ones, and they teach a model nothing but "big things fail"."""
        h = self.aabb_half_extents()
        return bool(np.sort(h)[:2].min() < jaw_half_width * 0.95)


class LiberoObjectLibrary:
    """Discovers LIBERO assets and merges them into a host MJCF scene."""

    def __init__(self, asset_root: str | Path | None = None) -> None:
        root = Path(asset_root) if asset_root else None
        if root is None:
            for cand in DEFAULT_ASSET_ROOTS:
                if Path(cand).is_dir():
                    root = Path(cand)
                    break
        if root is None or not root.is_dir():
            raise FileNotFoundError(
                "Could not find the LIBERO asset root. Pass asset_root=..., or check "
                f"{DEFAULT_ASSET_ROOTS}."
            )
        self.root = root

    # -- discovery -----------------------------------------------------------

    @lru_cache(maxsize=1)
    def catalogue(self) -> tuple[LiberoObject, ...]:
        """Every graspable object we can parse. Cached: parsing 150 XMLs is not free."""
        out: list[LiberoObject] = []
        for cat in GRASPABLE_CATEGORIES:
            for xml in sorted((self.root / cat).glob("*/*.xml")):
                try:
                    obj = self._parse(xml, cat)
                except Exception:  # noqa: BLE001 - a malformed asset must not kill the run
                    continue
                if obj is not None and obj.n_boxes > 0:
                    out.append(obj)
        return tuple(out)

    def graspable(self, jaw_half_width: float = 0.044) -> tuple[LiberoObject, ...]:
        return tuple(o for o in self.catalogue() if o.is_graspable(jaw_half_width))

    def _parse(self, xml: Path, category: str) -> LiberoObject | None:
        root = ET.parse(xml).getroot()
        body = root.find(".//body[@name='object']")
        if body is None:
            return None

        sizes, poss, quats = [], [], []
        friction, density = 0.95, 100.0
        for g in body.findall("geom"):
            if g.get("group") != "0" or g.get("type") != "box":
                continue  # group 0 is the collision decomposition; group 1 is the visual mesh
            sizes.append([float(v) for v in g.get("size", "0 0 0").split()])
            poss.append([float(v) for v in g.get("pos", "0 0 0").split()])
            q = g.get("quat", "1 0 0 0")
            quats.append([float(v) for v in q.split()])
            if g.get("friction"):
                friction = float(g.get("friction").split()[0])
            if g.get("density"):
                density = float(g.get("density"))

        if not sizes:
            return None
        return LiberoObject(
            name=xml.stem,
            category=category,
            xml_path=xml,
            box_sizes=np.asarray(sizes, dtype=float),
            box_pos=np.asarray(poss, dtype=float),
            box_quat=np.asarray(quats, dtype=float),
            friction=friction,
            density=density,
        )

    # -- MJCF generation -----------------------------------------------------

    @staticmethod
    def body_xml(
        obj: LiberoObject,
        *,
        friction: float | None = None,
        density: float | None = None,
        scale: float = 1.0,
        include_visual: bool = True,
    ) -> tuple[str, str]:
        """MJCF for the object: (asset_block, geoms_block).

        The visual mesh is referenced with an ABSOLUTE path so the host scene does not need a
        matching `meshdir`, and it is marked contype=0/conaffinity=0 so it is SEEN but never
        COLLIDED -- contact is resolved entirely against the convex decomposition, exactly as
        LIBERO itself does. That separation is deliberate and is the whole experiment: the camera
        sees the true shape, the physics collides a decomposition, and the analytic planner gets
        only a bounding box.
        """
        d = obj.xml_path.parent
        root = ET.parse(obj.xml_path).getroot()

        # HOIST THE OBJECT'S OWN <asset> BLOCK, rather than guessing where its mesh lives.
        #
        # Guessing does not work and fails quietly: only 8 of the 37 graspable LIBERO objects use
        # `visual/textured_vis.msh`; the rest reference their own `<name>.obj`. A guessed path that
        # does not exist simply produced no visual geom, and since the collision decomposition is
        # rendered with alpha = 0, the object came out INVISIBLE in the camera image while still
        # being perfectly solid to the physics. A dataset of empty photographs of objects that are
        # nonetheless there is a very effective way to train a model on nothing.
        asset_parts: list[str] = []
        mesh_map: dict[str, str] = {}
        mat_map: dict[str, str] = {}
        tex_names: dict[str, str] = {}

        src_asset = root.find("asset")
        if include_visual and src_asset is not None:
            for tx in src_asset.findall("texture"):
                fn = tx.get("file")
                if not fn:
                    continue
                nm = f"{obj.name}_{tx.get('name', 'tex')}"
                tex_names[tx.get("name", "")] = nm
                asset_parts.append(f'<texture name="{nm}" type="2d" file="{(d / fn).resolve()}"/>')
            for mt in src_asset.findall("material"):
                nm = f"{obj.name}_{mt.get('name', 'mat')}"
                mat_map[mt.get("name", "")] = nm
                tex = tex_names.get(mt.get("texture", ""), "")
                trefl = mt.get("reflectance", "0.15")
                asset_parts.append(
                    f'<material name="{nm}"'
                    + (f' texture="{tex}"' if tex else "")
                    + f' texrepeat="1 1" texuniform="false" reflectance="{trefl}"/>'
                )
            for ms in src_asset.findall("mesh"):
                fn = ms.get("file")
                if not fn:
                    continue
                nm = f"{obj.name}_{ms.get('name', 'mesh')}"
                mesh_map[ms.get("name", "")] = nm
                sc = [float(v) * scale for v in ms.get("scale", "1 1 1").split()]
                asset_parts.append(
                    f'<mesh name="{nm}" file="{(d / fn).resolve()}" '
                    f'scale="{sc[0]} {sc[1]} {sc[2]}"/>'
                )

        asset = "\n    ".join(asset_parts)

        # The visual geom is whatever the object's own body declares as group="1".
        vis_geom = ""
        src_body = root.find(".//body[@name='object']")
        if include_visual and src_body is not None:
            for g in src_body.findall("geom"):
                if g.get("group") != "1" or g.get("type") != "mesh":
                    continue
                mesh_nm = mesh_map.get(g.get("mesh", ""), "")
                if not mesh_nm:
                    continue
                mat_nm = mat_map.get(g.get("material", ""), "")
                style = f'material="{mat_nm}"' if mat_nm else 'rgba="0.72 0.66 0.58 1"'
                vis_geom = (
                    f'<geom name="{obj.name}_vis" type="mesh" mesh="{mesh_nm}" {style} '
                    f'group="1" contype="0" conaffinity="0" density="0"/>'
                )
                break

        if include_visual and not vis_geom:
            raise RuntimeError(
                f"{obj.name}: could not resolve a visual mesh from its MJCF. Rendering it would "
                "produce an INVISIBLE object that is nonetheless solid -- a photograph of nothing, "
                "with physics. Refusing rather than silently emitting an empty scene."
            )

        fr = obj.friction if friction is None else friction
        de = obj.density if density is None else density

        parts = [vis_geom] if vis_geom else []
        for i in range(obj.n_boxes):
            s = obj.box_sizes[i] * scale
            p = obj.box_pos[i] * scale
            q = obj.box_quat[i]
            parts.append(
                f'<geom name="objcol{i}" type="box" group="0" '
                f'size="{s[0]:.6f} {s[1]:.6f} {s[2]:.6f}" '
                f'pos="{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}" '
                f'quat="{q[0]:.6f} {q[1]:.6f} {q[2]:.6f} {q[3]:.6f}" '
                f'friction="{fr} 0.05 0.002" density="{de}" rgba="0.8 0.8 0.8 0"/>'
            )
        return asset, "\n      ".join(parts)
