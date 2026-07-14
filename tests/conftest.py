"""Test-session setup.

MuJoCo needs an offscreen GL backend on a headless machine. Without MUJOCO_GL=egl, importing the
renderer raises "an OpenGL platform library has not been loaded into this process" -- and it must be
set BEFORE mujoco creates a context, which is why it lives here rather than in a fixture.
"""

import os

os.environ.setdefault("MUJOCO_GL", "egl")
