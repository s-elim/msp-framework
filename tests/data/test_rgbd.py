"""The RGB-D observation kernel p(o | x). This is the module that makes MSP an experiment.

Before it, the "observation" the encoder consumed was a 32-dimensional random linear projection of
the state. The framework's central claim -- that a camera cannot resolve the world state, so
perception must output a calibrated BELIEF rather than a pose -- is not testable against a vector.
It is testable against an image.
"""

from __future__ import annotations

import os

import pytest
import torch

mujoco = pytest.importorskip("mujoco")
os.environ.setdefault("MUJOCO_GL", "egl")

from msp.oracle import AnalyticGraspOracle  # noqa: E402
from msp.oracle.mujoco_sim import MuJoCoOracle  # noqa: E402

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def scenes():
    an = AnalyticGraspOracle(shape="box")
    sim = MuJoCoOracle(shape="box")
    g = torch.Generator().manual_seed(0)
    x = an.sample_states(24, generator=g)
    return an, sim, x, sim.render(x, height=64, width=64, depth_noise=0.0)


def test_render_produces_a_valid_rgbd_tensor(scenes) -> None:
    _, _, x, obs = scenes
    assert obs.shape == (24, 4, 64, 64)
    assert obs.dtype == torch.float32
    assert torch.all(torch.isfinite(obs))
    assert 0.0 <= float(obs.min()) and float(obs.max()) <= 1.0, "channels must be normalized"


def test_the_camera_actually_sees_the_object(scenes) -> None:
    """A render that is the same regardless of the scene is a very expensive constant.

    Both the colour and the depth channel must vary with the state, or the encoder is being fed
    an image of nothing and any 'learning' it does is memorization of the action prior.
    """
    _, _, _, obs = scenes
    rgb_var = obs[:, :3].mean(dim=(1, 2, 3)).std()
    depth_var = obs[:, 3].std(dim=(1, 2)).mean()
    assert float(rgb_var) > 1e-3, f"RGB does not vary across scenes (std={float(rgb_var):.5f})"
    assert float(depth_var) > 1e-2, f"depth is flat within a frame (std={float(depth_var):.5f})"


def test_depth_is_a_real_depth_map(scenes) -> None:
    """The object must be NEARER than the table behind it. If depth were a constant or inverted,
    the encoder would still train and the loss would still fall -- silently, on garbage."""
    _, _, _, obs = scenes
    depth = obs[:, 3]
    # The nearest pixel in each frame (the object) must be meaningfully nearer than the median
    # (the table plane).
    nearest = depth.flatten(1).min(dim=1).values
    median = depth.flatten(1).median(dim=1).values
    assert torch.all(nearest < median), "nothing in the scene is nearer than the table"
    assert float((median - nearest).mean()) > 0.02


def test_the_camera_cannot_see_friction_mass_or_centre_of_mass(scenes) -> None:
    """THE SCIENTIFIC POINT OF THE WHOLE SETUP.

    Friction, mass and the centre of mass decide slip and torque failure, and NO PIXEL REVEALS
    THEM. Two scenes that differ only in those three variables must render identically -- and yet
    they must produce DIFFERENT grasp outcomes.

    That is exactly the irreducible epistemic uncertainty MSP's belief is built to carry, that a
    pose estimator cannot even express, and that active touch (Eq 20/21) exists to resolve. If this
    test ever fails, the experiment has been made artificially easy and the central claim is no
    longer being tested.
    """
    an, sim, x, obs = scenes

    x2 = x.clone()
    x2[:, 9] += 0.8  # log_friction
    x2[:, 10] += 1.2  # log_mass
    x2[:, 11:14] += 0.01  # centre of mass
    obs2 = sim.render(x2, height=64, width=64, depth_noise=0.0)

    torch.testing.assert_close(obs, obs2, atol=2e-2, rtol=0)

    # ...but the physics must NOT be invariant to them, or they would not matter at all.
    g = torch.Generator().manual_seed(1)
    a = an.sample_actions(x, 8, generator=g)
    assert not torch.allclose(an.outcome_params(x, a), an.outcome_params(x2, a)), (
        "friction/mass/COM changed nothing in the outcome -- the task has no hidden physics and "
        "there is nothing for active touch to discover"
    )


def test_depth_noise_is_applied(scenes) -> None:
    """A model trained on noiseless depth learns to trust it absolutely. Real sensors are not exact."""
    _, sim, x, _ = scenes
    clean = sim.render(x[:4], height=64, width=64, depth_noise=0.0)
    noisy = sim.render(x[:4], height=64, width=64, depth_noise=0.01)
    assert not torch.allclose(clean[:, 3], noisy[:, 3])
    torch.testing.assert_close(clean[:, :3], noisy[:, :3])  # RGB must be untouched


def test_rendering_reuses_one_gl_context(scenes) -> None:
    """PERFORMANCE REGRESSION. `_model` bakes size/mass/friction into the MJCF and so returns a
    distinct model per scene; a renderer is bound to a model, so a naive implementation builds a
    fresh GL context per frame. Context creation costs ~100 ms and turned a 2000-scene render into
    something that did not finish in ten minutes. We compile once and write geom_size directly."""
    _, sim, x, _ = scenes
    sim.render(x, height=64, width=64)
    assert len(sim._renderers) <= 2, (
        f"{len(sim._renderers)} GL contexts created for one batch; the renderer is being rebuilt "
        "per scene"
    )
