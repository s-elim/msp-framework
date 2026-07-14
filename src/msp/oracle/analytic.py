"""An analytic, differentiable grasp-outcome operator: the first tier of M.

Formalization Section 11 prescribes exactly this::

    M (train):  analytic Ferrari-Canny epsilon-quality + force-closure margin
                (differentiable prior) composed with a version-pinned rigid-body
                contact simulator for slip/perturbation.

This module is the differentiable prior. `MuJoCoOracle` is the simulator, and `CompositeOracle`
composes them. Together they replace the `torch.randint` that the audited repository shipped as
its physics.

WHY DIFFERENTIABILITY IS NOT OPTIONAL HERE. Contribution C2 obtains the outcome Jacobian J(x)
by autodiff through Phi (Section 11: "estimate J(x) by autodiff of Phi in sim"). An oracle that
can only be *sampled* forces J(x) onto finite differences, which in float32 is a coin toss (see
`diagnostics.outcome_jacobian`). So the quality metric here is built to be differentiable end to
end in the state, not merely evaluable.

THE PHYSICS

  State x = (pose[6], log_size[3], log_friction, log_mass, com[3])   -> d_X = 14
  Action a = (position[3], quaternion[4])                            -> gripper pose

  1. The object is a box with side lengths exp(log_size), at pose (t, R).
  2. A parallel-jaw gripper approaches along the action's local +z and closes along local +x.
     Jaw contacts are found by intersecting the closing axis with the box surface.
  3. At each contact we build a LINEARIZED friction cone with `cone_edges` generators.
  4. Each cone generator produces a wrench w = [f ; (p - com) x f] in R^6.
  5. The Ferrari-Canny epsilon quality is the radius of the largest origin-centred ball inside
     the convex hull of the wrench set. We evaluate its support-function form on a fixed set of
     D unit directions:

         eps  =  min_d  max_{i,j}  < w_ij , u_d >

     which is the standard L1 Ferrari-Canny metric discretized over directions. It is exactly
     differentiable (min and max have subgradients everywhere), it vectorizes over the batch,
     and eps > 0 iff the origin lies inside the hull, i.e. iff the grasp is force-closed.

  6. success = (eps > tau) AND the jaws actually contact the object.
     margin  = eps (the wrench-space stability margin).
     slip    = the tangential demand that the friction cone cannot absorb, under gravity.

None of this is a mesh, and none of it is a pose target. The supervision that leaves this module
is a wrench-space OUTCOME -- which is the whole reformulation.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from msp.oracle.base import PhysicsOracle
from msp.types import Outcome

__all__ = ["AnalyticGraspOracle", "STATE_DIM", "STATE_SLICES"]

STATE_DIM = 14
STATE_SLICES = {
    "pose_t": slice(0, 3),  # object translation
    "pose_r": slice(3, 6),  # object rotation, axis-angle
    "log_size": slice(6, 9),  # box half-extents, log
    "log_friction": slice(9, 10),  # Coulomb mu, log
    "log_mass": slice(10, 11),  # kg, log
    "com": slice(11, 14),  # centre of mass offset, in the object frame
}

_G = 9.81


def _axis_angle_to_matrix(v: Tensor) -> Tensor:
    """Rodrigues. (B, 3) -> (B, 3, 3). Differentiable, and safe at theta = 0."""
    theta = v.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    k = v / theta
    K = torch.zeros(*v.shape[:-1], 3, 3, device=v.device, dtype=v.dtype)
    K[..., 0, 1], K[..., 0, 2] = -k[..., 2], k[..., 1]
    K[..., 1, 0], K[..., 1, 2] = k[..., 2], -k[..., 0]
    K[..., 2, 0], K[..., 2, 1] = -k[..., 1], k[..., 0]
    I = torch.eye(3, device=v.device, dtype=v.dtype).expand_as(K)
    th = theta.unsqueeze(-1)
    return I + torch.sin(th) * K + (1 - torch.cos(th)) * (K @ K)


def _quat_to_matrix(q: Tensor) -> Tensor:
    """(..., 4) [w,x,y,z] -> (..., 3, 3). Normalizes first; the scale factor is derived from
    the NORMALIZED quaternion, which is the bug the audited geometry.py had."""
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    w, x, y, z = q.unbind(-1)
    return torch.stack(
        [
            1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
            2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
            2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y),
        ],
        dim=-1,
    ).reshape(*q.shape[:-1], 3, 3)


def _wrench_directions(n: int, device: torch.device, seed: int = 0) -> Tensor:
    """Unit directions in R^6 at which the support function of the wrench hull is evaluated.

    eps = min_d max_ij <w_ij, u_d> is an UPPER bound on the true inradius, and it is only as
    good as the direction set. Random directions alone are not enough, and the failure mode is
    exactly the one that matters:

    A two-finger grasp with POINT contacts has contacts at r1 = -r2 along the closing axis. Every
    torque r x f is then perpendicular to that axis, so every generator has an identically zero
    torsional component about it. The wrench hull lies inside a coordinate HYPERPLANE and the
    origin is on its boundary -- the grasp is not force-closed, and it must score eps <= 0. But
    the degenerate direction is a single axis of the 6-sphere, and 128 random directions miss it
    with near-certainty, so the metric happily reports force closure for a grasp that would
    unscrew itself in the hand.

    Including the 12 canonical axes +-e_i guarantees any hull that lies in a coordinate
    hyperplane is detected exactly. Fixed and seeded: a quality metric must not be stochastic,
    or two queries of the same (x, a) would return different "ground truth".
    """
    axes = torch.cat([torch.eye(6), -torch.eye(6)], dim=0)  # 12 canonical directions
    g = torch.Generator().manual_seed(seed)
    u = torch.randn(max(n - 12, 0), 6, generator=g)
    u = u / u.norm(dim=-1, keepdim=True)
    return torch.cat([axes, u], dim=0).to(device)


class AnalyticGraspOracle(PhysicsOracle):
    """Ferrari-Canny epsilon-quality on a box, differentiable in the full state.

    Args:
        cone_edges: generators of the linearized friction cone at each contact. 8 is standard.
        n_directions: unit wrench directions used to evaluate the inradius. More is a tighter
            (larger) epsilon; 128 is a good speed/fidelity point.
        eps_threshold: tau. A grasp succeeds when its epsilon quality exceeds it.
        noise: sampling noise for `query`. `outcome_params` is always noiseless -- J(x) is the
            Jacobian of the MEAN outcome map, not of a sample.
    """

    differentiable = True

    # Gripper geometry. These MUST mirror the MJCF in mujoco_sim.py -- the two tiers are only
    # comparable if they model the same hand.
    JAW_INNER_HALF_WIDTH: float = 0.044  # open jaws reach +-0.044 about the closing axis
    FINGER_BELOW_TCP: float = 0.020  # fingertips extend this far below the TCP
    PALM_ABOVE_TCP: float = 0.048  # palm underside sits this far above the TCP

    def __init__(
        self,
        shape: str = "box",  # "box" | "cylinder"
        cone_edges: int = 8,
        n_directions: int = 128,
        eps_threshold: float = 0.02,
        torsional_coeff: float = 0.02,  # gamma: soft-finger contact-patch radius, metres
        gripper_width: float = 0.085,  # Franka Panda, metres
        noise: bool = True,
        device: torch.device | str = "cpu",
    ) -> None:
        if shape not in ("box", "cylinder"):
            raise ValueError(f"shape must be 'box' or 'cylinder'; got {shape!r}")
        self.shape = shape
        self.cone_edges = cone_edges
        self.eps_threshold = eps_threshold
        self.torsional_coeff = torsional_coeff
        self.gripper_width = gripper_width
        self.noise = noise
        self.device = torch.device(device)
        self.directions = _wrench_directions(n_directions, self.device)
        #: (B, 3) per-scene base half-extents, or None for the parametric box/cylinder world.
        self.base_half: Tensor | None = None

    @property
    def state_dim(self) -> int:
        return STATE_DIM

    def to(self, device: torch.device | str) -> AnalyticGraspOracle:
        self.device = torch.device(device)
        self.directions = self.directions.to(self.device)
        return self

    # -- geometry ------------------------------------------------------------

    def _half(self, state: Tensor) -> Tensor:
        """The half-extents the ANALYTIC tier plans on. (B, 3).

        With `base_half` set, this is a per-scene ORIENTED BOUNDING BOX -- a deliberately crude
        stand-in for the object's true shape, of about the fidelity a pose-and-shape pipeline
        produces. The simulator meanwhile collides the object's true convex decomposition. The
        disagreement between the two is not a nuisance to be minimized; it is the measurement the
        paper is about (blueprint wrong-assumption #11: force closure evaluated on a hallucinated
        surface passes the check and fails the lift). On a box that gap is identically zero, which
        is precisely why boxes cannot test the claim.

        Without `base_half` (the synthetic box/cylinder world), the state's log_size IS the size.
        """
        if self.base_half is None:
            return torch.exp(state[:, STATE_SLICES["log_size"]]).clamp(0.018, 0.038)

        b = state.shape[0]
        base = self.base_half.to(state.device)
        if base.shape[0] == 1:
            base = base.expand(b, 3)
        elif base.shape[0] != b:
            # `outcome_jacobian` evaluates Phi at a SINGLE state, so a base_half installed for a
            # whole batch will not match. Say so, rather than letting a reshape fail three frames
            # deep inside the wrench metric.
            raise ValueError(
                f"base_half has {base.shape[0]} scenes but {b} states were passed. Install the "
                "bounding box for exactly the scenes you are about to evaluate: "
                "oracle.set_base_half(sim.base_half(object_index[i:i+1])) for a single state."
            )
        scale = torch.exp(state[:, STATE_SLICES["log_size"]])  # (B, 3), nominal 1.0
        return (base * scale).clamp(0.008, 0.060)

    def set_base_half(self, base_half: Tensor | None) -> AnalyticGraspOracle:
        """Install the per-scene bounding boxes for the current batch. Returns self."""
        self.base_half = base_half
        return self

    def _contacts(
        self, state: Tensor, actions: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Where the two jaws touch the box, and with what inward normals.

        Returns (points, normals, in_contact, half) with shapes
        (B, Na, 2, 3), (B, Na, 2, 3), (B, Na), (B, 3).
        """
        b = state.shape[0]
        na = actions.shape[1]

        t_obj = state[:, STATE_SLICES["pose_t"]]  # (B, 3)
        R_obj = _axis_angle_to_matrix(state[:, STATE_SLICES["pose_r"]])  # (B, 3, 3)
        half = self._half(state)

        g_t = actions[..., 0:3]  # (B, Na, 3)
        R_g = _quat_to_matrix(actions[..., 3:7])  # (B, Na, 3, 3)
        close_axis = R_g[..., :, 0]  # local +x = the closing direction, (B, Na, 3)

        # Work in the OBJECT frame: the box is then axis-aligned and the ray/box intersection
        # is a slab test, which is exact and differentiable.
        Rt = R_obj.transpose(-1, -2).unsqueeze(1)  # (B, 1, 3, 3)
        o = torch.einsum("bnij,bnj->bni", Rt.expand(b, na, 3, 3), g_t - t_obj.unsqueeze(1))
        d = torch.einsum("bnij,bnj->bni", Rt.expand(b, na, 3, 3), close_axis)
        d = d / d.norm(dim=-1, keepdim=True).clamp_min(1e-8)

        if self.shape == "cylinder":
            # A cylinder is invariant under rotation about its own z axis. We make that
            # invariance EXACT rather than approximate by expressing the ray in the
            # cylinder's rotationally-symmetric coordinates: only the radial distance from
            # the axis matters, never the azimuth. The x and y half-extents are tied to a
            # single radius for the same reason.
            half = torch.stack([half[:, 0], half[:, 0], half[:, 2]], dim=-1)

        h = half.unsqueeze(1)  # (B, 1, 3)
        inv_d = 1.0 / torch.where(d.abs() < 1e-6, torch.full_like(d, 1e-6), d)
        t1 = (-h - o) * inv_d
        t2 = (h - o) * inv_d
        t_near = torch.minimum(t1, t2).max(dim=-1).values  # (B, Na)
        t_far = torch.maximum(t1, t2).min(dim=-1).values

        hit = (t_far > t_near) & ((t_far - t_near) < self.gripper_width)

        p_a = o + t_near.unsqueeze(-1) * d  # (B, Na, 3), object frame
        p_b = o + t_far.unsqueeze(-1) * d

        # INWARD surface normal of an axis-aligned box.
        #
        # For any point p on the surface, the inward normal points back toward the centre, so it
        # is simply -sign(p) on the axis the point is saturated on. It does NOT depend on which
        # jaw produced the point. An earlier version took a `sign` argument and flipped the
        # normal for the far contact, which made BOTH normals point the same way: n1 . n2 = +1
        # instead of -1. The grasp was therefore never antipodal, force closure was almost
        # unreachable (it needed mu > 1.0 on a perfect cube), and the success label was
        # degenerate. The whole wrench-space metric was being fed garbage geometry.
        def inward_normal(p: Tensor) -> Tensor:
            r = p / h.clamp_min(1e-6)
            # Soft one-hot on the dominant axis, so the normal stays differentiable in the pose.
            w = torch.softmax(r.abs() * 40.0, dim=-1)
            return -w * torch.sign(r)

        n_a = inward_normal(p_a)
        n_b = inward_normal(p_b)

        # Back to the world frame.
        Rw = R_obj.unsqueeze(1)  # (B, 1, 3, 3)
        to_world = lambda v: torch.einsum("bnij,bnj->bni", Rw.expand(b, na, 3, 3), v)  # noqa: E731
        pts = torch.stack([to_world(p_a), to_world(p_b)], dim=2)  # (B, Na, 2, 3)
        nrm = torch.stack([to_world(n_a), to_world(n_b)], dim=2)
        nrm = nrm / nrm.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        return pts, nrm, hit, half

    # -- wrench space --------------------------------------------------------

    def _epsilon_quality(self, state: Tensor, actions: Tensor) -> tuple[Tensor, Tensor]:
        """Ferrari-Canny epsilon and the tangential slip demand. Both (B, Na)."""
        pts, nrm, hit, half = self._contacts(state, actions)
        b, na = hit.shape

        mu = torch.exp(state[:, STATE_SLICES["log_friction"]]).clamp(0.05, 2.0)  # (B, 1)
        mass = torch.exp(state[:, STATE_SLICES["log_mass"]]).clamp(0.01, 10.0)  # (B, 1)
        com = state[:, STATE_SLICES["com"]]  # (B, 3)

        # Tangent basis at each contact.
        n = nrm  # (B, Na, 2, 3)
        helper = torch.zeros_like(n)
        helper[..., 0] = 1.0
        alt = torch.zeros_like(n)
        alt[..., 1] = 1.0
        helper = torch.where((n[..., 0:1].abs() > 0.9), alt, helper)
        t1 = torch.cross(n, helper, dim=-1)
        t1 = t1 / t1.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        t2 = torch.cross(n, t1, dim=-1)

        # Linearized friction cone generators: f_ij = n + mu*(cos a * t1 + sin a * t2)
        ang = torch.arange(self.cone_edges, device=state.device, dtype=state.dtype)
        ang = ang * (2 * math.pi / self.cone_edges)
        c, s = torch.cos(ang), torch.sin(ang)  # (E,)
        m = mu.view(b, 1, 1, 1, 1)
        f = (
            n.unsqueeze(3)
            + m * (c.view(1, 1, 1, -1, 1) * t1.unsqueeze(3)
                   + s.view(1, 1, 1, -1, 1) * t2.unsqueeze(3))
        )  # (B, Na, 2, E, 3)
        f = f / f.norm(dim=-1, keepdim=True).clamp_min(1e-8)

        # Wrench of each generator about the centre of mass.
        #
        # TORQUE SCALING -- without this the metric is meaningless, and the failure is subtle.
        # A wrench is [force ; torque], but the two have different UNITS. Forces here are unit
        # vectors, O(1). Torques are r x f with |r| ~ 0.03 m, so they are O(0.03). The unit
        # directions u_d are drawn uniformly on the sphere in R^6 and therefore weight the force
        # and torque subspaces EQUALLY, so `min_d <w, u_d>` is dominated by the torque directions,
        # where the support is ~30x smaller. Epsilon then comes out negative for essentially every
        # grasp regardless of how good it is, and the success label collapses to a constant.
        #
        # The standard fix (Ferrari & Canny) is to divide torque by a characteristic length
        # lambda -- the object's radius -- which makes the two subspaces commensurate and the
        # epsilon ball meaningful. Reported epsilon values are only comparable across objects
        # BECAUSE of this normalization.
        lam = half.norm(dim=-1).clamp_min(1e-3).view(b, 1, 1, 1, 1)  # (B,1,1,1,1), metres

        r = pts.unsqueeze(3) - com.view(b, 1, 1, 1, 3)  # (B, Na, 2, E, 3)
        tau = torch.cross(r, f, dim=-1) / lam

        # SOFT-FINGER CONTACT -- and this is physics, not a fudge factor.
        #
        # A two-finger grasp with ideal POINT contacts is provably NOT force-closed in the full
        # 6D wrench space: two contacts generate at most 2E wrench generators, and no positive
        # combination of them can resist a torsion about the line joining the two contact points.
        # So epsilon <= 0 for essentially every grasp, the success label collapses to a constant
        # 0, and Assumption A8 (informative outcomes) fails outright -- the dataset would carry
        # no information about x at all.
        #
        # Real fingertips are not points. They make a contact PATCH that resists a torsional
        # moment up to gamma * mu * f_n about the contact normal (the soft-finger model of
        # Howe & Cutkosky). `torsional_coeff` is gamma, the effective patch radius in metres.
        # Adding it is what makes a parallel-jaw grasp force-closable, which is the empirical
        # fact that parallel-jaw grippers work.
        torsion = (self.torsional_coeff / lam) * m * n.unsqueeze(3)  # (B, Na, 2, E, 3)
        w = torch.cat(
            [
                torch.cat([f, tau + torsion], dim=-1),  # +tau_z generator
                torch.cat([f, tau - torsion], dim=-1),  # -tau_z generator
            ],
            dim=3,
        )  # (B, Na, 2, 2E, 6)
        w = w.reshape(b, na, 4 * self.cone_edges, 6)

        # eps = min_d  max_ij  <w_ij, u_d>   -- the inradius of the wrench hull.
        proj = torch.einsum("bnkc,dc->bnkd", w, self.directions)  # (B, Na, K, D)
        support = proj.max(dim=2).values  # (B, Na, D)
        eps = support.min(dim=-1).values  # (B, Na)

        # A grasp that never touched the object has no quality at all.
        eps = torch.where(hit, eps, torch.full_like(eps, -1.0))

        # Slip: the gravity wrench must be absorbed inside the friction cone. The tangential
        # demand beyond what mu can hold is what actually slips.
        weight = (mass * _G).view(b, 1)  # (B, 1)
        gdir = torch.tensor([0.0, 0.0, -1.0], device=state.device, dtype=state.dtype)
        n_dot_g = (n * gdir).sum(-1)  # (B, Na, 2) -- normal component available
        normal_cap = mu.view(b, 1) * n_dot_g.abs().sum(dim=-1).clamp_min(1e-3)  # (B, Na)
        demand = weight / normal_cap.clamp_min(1e-3)
        slip = torch.relu(demand - 1.0) * 0.01  # metres of post-lift deviation
        slip = torch.where(hit, slip, torch.full_like(slip, 0.05))
        return eps, slip

    # -- the operator --------------------------------------------------------

    def _reachability(self, state: Tensor, actions: Tensor) -> Tensor:
        """Signed clearance of the OPEN gripper against the object and the table. (B, Na).

        Positive means the grasp pose is executable; negative means the jaws, or the palm, or the
        fingertips are already inside something before the grasp even starts.

        WHY THIS BELONGS IN THE ANALYTIC TIER. Force closure asks "if the jaws were in contact
        here, would the grasp hold?" It says nothing about whether the gripper can BE there. Without
        a reachability term the analytic tier claimed a 96% success rate against a simulator that
        measured 33%, because it was scoring grasps that drive the fingers through the table. That
        is not a hard grasp; it is an impossible one, and a prior that cannot tell the difference is
        a bad prior. This term is the geometric half of "can this action be executed".

        The three clearances mirror the gripper's geometry exactly (see mujoco_sim.TCP_OFFSET):

            jaw    the object's extent along the CLOSING axis must fit inside the open jaws
            floor  the fingertips, 0.020 below the TCP, must stay above the table
            palm   the palm, 0.048 above the TCP, must stay above the top of the object
        """
        b, na, _ = actions.shape
        R_obj = _axis_angle_to_matrix(state[:, STATE_SLICES["pose_r"]])  # (B,3,3)
        half = self._half(state)  # (B, 3)

        R_g = _quat_to_matrix(actions[..., 3:7])  # (B,Na,3,3)
        close_axis = R_g[..., :, 0]  # local +x, world frame
        tcp = actions[..., 0:3]  # (B,Na,3), relative to the object's centre

        # Jaw clearance: the object's extent along the CLOSING axis, in the OBJECT frame.
        c_obj = torch.einsum("bji,bnj->bni", R_obj, close_axis)  # world -> object frame

        if self.shape == "cylinder":
            # THE SUPPORT FUNCTION MUST RESPECT THE SYMMETRY, or the symmetry is not there.
            #
            # A cylinder's support along direction c is  r*sqrt(cx^2 + cy^2) + h*|cz|, which depends
            # on c only through its radial magnitude -- rotationally invariant about the axis, as it
            # must be. Using the BOX support sum_i |c_i| h_i here instead is wrong and not harmlessly
            # so: |cos t| + |sin t| varies with the azimuth t, so it makes grasp outcomes depend on
            # the cylinder's yaw. That silently destroys the very outcome-invariance Theorem 4 is
            # about, and `dim ker J` collapsed from 1 to 0 -- the framework's headline identifiability
            # result, deleted by a support function copied from the wrong shape.
            radial = c_obj[..., :2].norm(dim=-1)
            extent = radial * half[:, 0].unsqueeze(1) + c_obj[..., 2].abs() * half[:, 2].unsqueeze(1)
        else:
            extent = (c_obj.abs() * half.unsqueeze(1)).sum(-1)  # box: sum_i |c_i| h_i

        jaw = self.JAW_INNER_HALF_WIDTH - extent

        # Floor and palm clearance, in WORLD z.
        #
        # The relevant height is the object's extent along world +z, which for a rotated object is
        # the support function sum_i |R[2,i]| * half_i -- NOT its object-frame z half-extent. A
        # bottle lying on its side is short in the world and tall in its own frame, and using the
        # wrong one puts the palm clearance test on the wrong object entirely.
        world_half_z = (R_obj[:, 2, :].abs() * half).sum(dim=-1, keepdim=True)  # (B, 1)
        floor = (tcp[..., 2] - self.FINGER_BELOW_TCP) - (-world_half_z)
        palm = (tcp[..., 2] + self.PALM_ABOVE_TCP) - world_half_z

        return torch.minimum(torch.minimum(jaw, floor), palm)

    def outcome_params(self, state: Tensor, actions: Tensor) -> Tensor:
        """Phi(x) = (succ_logit, margin, log_slip) per action, flattened. Differentiable."""
        eps, slip = self._epsilon_quality(state, actions)
        clear = self._reachability(state, actions)

        # A grasp succeeds if it is BOTH force-closed AND executable. Both terms are smooth, so
        # Phi stays differentiable and J(x) is still available by autodiff.
        quality_logit = 60.0 * (eps - self.eps_threshold)
        reach_logit = 300.0 * clear
        succ_logit = torch.minimum(quality_logit, reach_logit)

        margin = eps
        log_slip = torch.log(slip.clamp_min(1e-4))
        stacked = torch.stack([succ_logit, margin, log_slip], dim=-1)  # (B, Na, 3)
        return stacked.reshape(state.shape[0], -1)

    @torch.no_grad()
    def query(self, state: Tensor, actions: Tensor) -> Outcome:
        eps, slip = self._epsilon_quality(state, actions)
        clear = self._reachability(state, actions)
        # Same success rule as `outcome_params`: force-closed AND executable. If the two ever
        # disagree, the Jacobian would be the Jacobian of a different operator than the one that
        # produced the labels, and C2 would be measuring a fiction.
        logit = torch.minimum(60.0 * (eps - self.eps_threshold), 300.0 * clear)
        p = torch.sigmoid(logit)
        slip = torch.where(clear < 0, torch.full_like(slip, 0.5), slip)

        if self.noise:
            succ = torch.bernoulli(p)
            margin = eps + 0.005 * torch.randn_like(eps)
            slip = (slip * torch.exp(0.1 * torch.randn_like(slip))).clamp_min(0.0)
        else:
            succ = (p > 0.5).float()
            margin = eps

        slip = slip * (1.0 - succ) + 1e-4 * succ  # a successful lift does not slip
        out = Outcome(
            succ=succ.unsqueeze(-1), margin=margin.unsqueeze(-1), slip=slip.unsqueeze(-1)
        )
        out.validate()
        return out

    # -- scene sampling ------------------------------------------------------

    def sample_states(self, n: int, generator: torch.Generator | None = None) -> Tensor:
        """Draw x ~ p(x): boxes of varying size, pose, friction, mass and COM offset."""
        g = generator
        x = torch.zeros(n, STATE_DIM)
        x[:, STATE_SLICES["pose_t"]] = torch.randn(n, 3, generator=g) * 0.02

        # YAW ONLY. An object resting on a table cannot be tilted: it lies flat on a face. Sampling
        # a free 3-DoF rotation produced boxes tipped by up to 40 degrees while still being placed
        # at their UNROTATED resting height, so they started the episode interpenetrating the table
        # -- and the gripper, aimed at a pose the object was not actually in, drove its fingers 10 mm
        # into the object. This is a scene-physics error masquerading as a grasping failure.
        x[:, STATE_SLICES["pose_r"]] = 0.0
        x[:, 5] = (torch.rand(n, generator=g) * 2 - 1) * math.pi  # yaw about world z
        x[:, STATE_SLICES["log_size"]] = math.log(0.028) + torch.randn(n, 3, generator=g) * 0.15
        x[:, STATE_SLICES["log_friction"]] = math.log(0.6) + torch.randn(n, 1, generator=g) * 0.3
        x[:, STATE_SLICES["log_mass"]] = math.log(0.3) + torch.randn(n, 1, generator=g) * 0.5
        x[:, STATE_SLICES["com"]] = torch.randn(n, 3, generator=g) * 0.004
        return x.to(self.device)

    def sample_actions(
        self,
        state: Tensor,
        n_actions: int,
        generator: torch.Generator | None = None,
        spread: float = 1.0,
    ) -> Tensor:
        """rho: grasp poses aimed near the object, with random approach orientations.

        Full support on the admissible set (Assumption A3), but concentrated where grasps are
        plausible -- a uniform prior over SE(3) would put essentially all its mass on grasps
        that miss the object entirely and carry no information.

        THE TRAINING PROPOSAL AND THE DEPLOYMENT PROPOSAL ARE NOT THE SAME DISTRIBUTION, and
        conflating them makes the conformal certificate vacuous.

        `spread` scales how far a proposal may stray from the ideal antipodal grasp.

        * spread = 1.0 (default, and what the CORPUS must use). Wide. About 10% of grasps succeed.
          Assumptions A3/A8 require this: a corpus of only good grasps carries no information about
          what makes a grasp fail, and the outcome head would have no negatives to learn from.
        * spread << 1. Tight. The candidate set a DEPLOYED system would actually rank.

        Why this knob has to exist. Under the wide proposal the best grasp out of 16000 succeeds
        about 52% of the time, and s -- which is calibrated, and says 0.46 for exactly those grasps
        -- never crosses 0.5. Eq 24 certifies only when s > max(q_hat, 1-q_hat) >= 0.5, so NOTHING is
        ever certified and the scene-level abstention rate is 1.000 at every alpha. That is not a
        broken certificate; it is an honest one being asked to certify a coin flip. A deployed
        system does not rank 16000 random grasps, it ranks good candidates, and with a tighter
        proposal the certificate has something it can actually certify.
        """
        b = state.shape[0]
        dev = state.device
        t_obj = state[:, STATE_SLICES["pose_t"]]
        R_obj = _axis_angle_to_matrix(state[:, STATE_SLICES["pose_r"]])

        # CHOOSE THE CLOSING AXIS IN THE WORLD FRAME, NOT THE OBJECT'S.
        #
        # A uniform prior over SE(3) would put essentially all of its mass on grasps that miss the
        # object entirely -- the labels would be a constant 0 and Assumption A8 (informative
        # outcomes) would fail. So the closing axis is aimed at one of the object's own principal
        # axes. But WHICH of them is graspable depends on how the object is LYING, and that is a
        # fact about the world frame, not the object frame.
        #
        # A top-down parallel jaw can only close across a roughly HORIZONTAL axis: it approaches
        # downward, so it cannot pinch an object along the vertical. And it should close across the
        # NARROW dimension -- you grasp a milk carton across its width, not along its length.
        #
        # Selecting the two narrowest axes in the OBJECT frame gets this right for an upright box and
        # wrong for everything else. A scanned bottle settles on its SIDE: its object-frame "short"
        # axis is then pointing straight up, the sampler aims the jaws at the vertical, the gripper
        # cannot close on it, and the grasp fails. Measured on the LIBERO groceries, that alone held
        # the simulated success rate at 5.5% while the analytic tier -- planning in the same broken
        # frame -- happily reported 82%, which would have been written up as a spectacular
        # hallucinated-surface gap rather than as a bug in the sampler.
        #
        # So: rank the object's axes by how HORIZONTAL they are in the world after settling, keep
        # the horizontal ones, and among those prefer the narrow ones.
        half = self._half(state)  # (B, 3) object-frame half-extents
        obj_axes = R_obj  # (B, 3, 3): column i is object axis i in world coordinates
        verticality = obj_axes[:, 2, :].abs()  # (B, 3) |axis . world_z|

        # Score: low is good. Penalize verticality hard (a vertical axis is ungraspable from above),
        # then prefer narrow axes. Normalizing the width by the largest extent keeps the two terms
        # commensurate across objects of very different size.
        width = half / half.max(dim=-1, keepdim=True).values.clamp_min(1e-6)
        score = 4.0 * verticality + width  # (B, 3)
        order = score.argsort(dim=-1)  # ascending: order[:, 0] is the best closing axis

        # Which closing axis. At full spread, either of the two best; as spread -> 0, always the
        # best one. The second-best axis is a legitimate grasp on a cube and a poor one on a carton,
        # so a tight proposal should stop gambling on it.
        take_second = torch.rand(b, n_actions, generator=generator, device=dev) < 0.5 * spread
        pick = take_second.long()
        axis_id = torch.gather(order, 1, pick)  # one of the two best axes
        axes = torch.eye(3, device=dev)[axis_id]  # (B, Na, 3), object frame
        axes = torch.einsum("bij,bnj->bni", R_obj, axes)  # -> world frame
        axes = axes + spread * 0.12 * torch.randn(
            b, n_actions, 3, generator=generator, device=dev
        )
        # Project the closing axis onto the horizontal plane: whatever the object's axes are, the
        # jaws of a top-down gripper close horizontally.
        axes[..., 2] = axes[..., 2] * 0.15
        axes = axes / axes.norm(dim=-1, keepdim=True).clamp_min(1e-8)

        # Build the gripper frame. Local +x is the closing axis; the remaining roll about it is
        # NOT free, and choosing it arbitrarily is a mistake with real consequences.
        #
        # The gripper reaches along its local -z (the fingers hang below the palm). If the roll is
        # picked at random, that approach direction points sideways or straight up as often as
        # down, so the hand is placed inside the table or reaching out of it, and the grasp is not
        # executable at all. Measured against the simulator, an arbitrary roll had ~96% of grasps
        # rejected for collision at placement.
        #
        # So we pick the roll that makes the approach as close to TOP-DOWN as the closing axis
        # allows: local +z is the component of world +z orthogonal to the closing axis. This is a
        # top-down antipodal grasp, which is what a table-top parallel-jaw gripper actually does.
        world_z = torch.zeros_like(axes)
        world_z[..., 2] = 1.0
        z_ax = world_z - (world_z * axes).sum(-1, keepdim=True) * axes
        # If the closing axis IS vertical, any roll is equivalent; fall back to world +x.
        degenerate = z_ax.norm(dim=-1, keepdim=True) < 1e-3
        fallback = torch.zeros_like(axes)
        fallback[..., 0] = 1.0
        z_ax = torch.where(degenerate, fallback, z_ax)
        z_ax = z_ax / z_ax.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        y_ax = torch.cross(z_ax, axes, dim=-1)
        y_ax = y_ax / y_ax.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        R = torch.stack([axes, y_ax, z_ax], dim=-1)  # (B, Na, 3, 3), columns

        tr = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
        w = torch.sqrt((1 + tr).clamp_min(1e-8)) / 2
        qx = (R[..., 2, 1] - R[..., 1, 2]) / (4 * w.clamp_min(1e-6))
        qy = (R[..., 0, 2] - R[..., 2, 0]) / (4 * w.clamp_min(1e-6))
        qz = (R[..., 1, 0] - R[..., 0, 1]) / (4 * w.clamp_min(1e-6))
        q = torch.stack([w, qx, qy, qz], dim=-1)
        q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-8)

        # Grasp the UPPER PART of the object, measured in the WORLD.
        #
        # A table-top gripper straddles the top of an object; aiming the tool centre point at the
        # object's CENTRE drives the fingertips below the table for anything short, and the grasp is
        # rejected for hitting the floor before it is ever simulated.
        #
        # The height must be the object's extent along WORLD +z, which for a rotated box is the
        # support function  h_z = sum_i |R[2,i]| * half_i  -- not the object-frame half-extent along
        # its own z. Using the latter is right only for an upright object and badly wrong for a
        # bottle lying on its side, where the object's "height" is now its width.
        half_obj = self._half(state)  # (B, 3)
        world_half_z = (R_obj[:, 2, :].abs() * half_obj).sum(dim=-1)  # (B,)

        lift = torch.zeros(b, n_actions, 3, device=dev)
        lift[..., 2] = 0.45 * world_half_z.unsqueeze(1)
        pos = (
            t_obj.unsqueeze(1)
            + lift
            + spread * 0.006 * torch.randn(b, n_actions, 3, generator=generator, device=dev)
        )
        return torch.cat([pos, q], dim=-1)

    def observe(self, state: Tensor, obs_dim: int = 64, noise: float = 0.01) -> Tensor:
        """A stand-in sensor. Sees pose and size; does NOT see friction, mass or COM.

        This is the physically honest part: those three are the variables that decide slip and
        torque failures, and no camera observes them. The belief must therefore carry genuine
        epistemic uncertainty about them -- which is exactly what active touch (Eq 20/21) is
        for, and what a pose estimator cannot express at all.
        """
        visible = torch.cat(
            [state[:, 0:9], torch.zeros_like(state[:, 9:])], dim=-1
        )  # friction/mass/com zeroed out
        g = torch.Generator().manual_seed(7)
        A = (torch.randn(STATE_DIM, obs_dim, generator=g) / STATE_DIM**0.5).to(state.device)
        return visible @ A + noise * torch.randn(
            state.shape[0], obs_dim, device=state.device
        )
