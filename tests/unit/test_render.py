from typing import Any

import mujoco
import numpy as np
import pytest
from conftest import skip_if_headless

from crazyflow import Sim


class _FakeCamera:
    def __init__(self) -> None:
        self.fixedcamid = -1
        self.type = mujoco.mjtCamera.mjCAMERA_FREE


class _FakeViewer:
    def __init__(self) -> None:
        self.cam = _FakeCamera()


class _FakeRenderer:
    instances: list["_FakeRenderer"] = []

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.camera_id = kwargs["camera_id"]
        self.width = kwargs["width"]
        self.height = kwargs["height"]
        self.cam_config = kwargs["default_cam_config"]
        self.viewer: _FakeViewer | None = None
        self._viewers: dict[str | None, _FakeViewer] = {}
        self.camera_at_render: list[tuple[str | None, int, int]] = []
        self.closed = False
        self.instances.append(self)

    def _get_viewer(self, render_mode: str | None) -> _FakeViewer:
        self.viewer = self._viewers.get(render_mode)
        if self.viewer is None:
            self.viewer = _FakeViewer()
            for key, value in (self.cam_config or {}).items():
                setattr(self.viewer.cam, key, value)
            self._viewers[render_mode] = self.viewer
        return self.viewer

    def render(self, mode: str | None) -> None:
        self._get_viewer(mode)
        assert self.viewer is not None
        self.camera_at_render.append((mode, self.viewer.cam.fixedcamid, self.viewer.cam.type))

    def close(self) -> None:
        self.closed = True


@pytest.fixture
def fake_renderer(monkeypatch: pytest.MonkeyPatch) -> type[_FakeRenderer]:
    _FakeRenderer.instances.clear()
    monkeypatch.setattr("crazyflow.sim.sim.MujocoRenderer", _FakeRenderer)
    return _FakeRenderer


@pytest.mark.unit
def test_render_rejects_changed_persistent_configuration(fake_renderer: type[_FakeRenderer]):
    sim = Sim(n_drones=1)
    camera_name = "fpv_cam:0"
    camera_id = mujoco.mj_name2id(sim.mj_model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
    cam_config = {"distance": 2.0, "lookat": [0.0, 0.0, 1.0]}
    sim.render(mode="rgb_array", camera=camera_name, width=320, height=240, cam_config=cam_config)

    # Equivalent names/IDs and list/array values reuse the original renderer.
    sim.render(
        mode="depth_array",
        camera=camera_id,
        width=320,
        height=240,
        cam_config={"distance": 2.0, "lookat": np.array([0.0, 0.0, 1.0])},
    )
    assert len(fake_renderer.instances) == 1

    incompatible = (
        {"camera": "track_cam:0", "width": 320, "height": 240, "cam_config": cam_config},
        {"camera": camera_name, "width": 640, "height": 240, "cam_config": cam_config},
        {"camera": camera_name, "width": 320, "height": 480, "cam_config": cam_config},
        {
            "camera": camera_name,
            "width": 320,
            "height": 240,
            "cam_config": {"distance": 3.0, "lookat": [0.0, 0.0, 1.0]},
        },
    )
    for settings in incompatible:
        with pytest.raises(ValueError, match=r"Call sim\.close\(\)"):
            sim.render(mode="rgb_array", **settings)

    assert len(fake_renderer.instances) == 1
    sim.close()


@pytest.mark.unit
def test_close_allows_renderer_reconfiguration(fake_renderer: type[_FakeRenderer]):
    sim = Sim(n_drones=1)
    sim.render(mode="rgb_array", camera="fpv_cam:0", width=320, height=240)
    first = fake_renderer.instances[-1]

    sim.close()
    assert first.closed
    assert sim.viewer is None

    sim.render(mode="rgb_array", camera="track_cam:0", width=640, height=480)
    second = fake_renderer.instances[-1]
    assert second is not first
    assert (second.width, second.height) == (640, 480)
    sim.close()


@pytest.mark.unit
def test_fixed_human_camera_after_offscreen_render(fake_renderer: type[_FakeRenderer]):
    sim = Sim(n_drones=1)
    camera_id = mujoco.mj_name2id(sim.mj_model, mujoco.mjtObj.mjOBJ_CAMERA, "fpv_cam:0")
    sim.render(mode="rgb_array", camera=camera_id, width=320, height=240)
    sim.render(mode="human", camera=camera_id, width=320, height=240)

    assert sim.viewer.viewer.cam.type == mujoco.mjtCamera.mjCAMERA_FIXED
    assert sim.viewer.viewer.cam.fixedcamid == camera_id
    human_draws = [entry for entry in sim.viewer.camera_at_render if entry[0] == "human"]
    assert human_draws == [("human", camera_id, mujoco.mjtCamera.mjCAMERA_FIXED)]
    sim.close()


@pytest.mark.unit
@pytest.mark.parametrize("cam_name", ["fpv_cam:0", "track_cam:0", "fpv_cam:1", "track_cam:1"])
@pytest.mark.render
@skip_if_headless
def test_render_camera_selection_from_name(cam_name: str):
    sim = Sim(drone="cf21B_500", n_drones=2)
    cam_id = mujoco.mj_name2id(sim.mj_model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
    sim.render(mode="human", camera=cam_name)
    viewer_cam = sim.viewer.viewer.cam
    assert viewer_cam.type == mujoco.mjtCamera.mjCAMERA_FIXED, "Camera type was not set to FIXED"
    assert viewer_cam.fixedcamid == cam_id, f"Expected cam ID {cam_id}, got {viewer_cam.fixedcamid}"
    sim.close()


@pytest.mark.unit
@pytest.mark.parametrize("cam_id", [0, 1, 2, 3])
@pytest.mark.render
@skip_if_headless
def test_render_camera_selection_from_id(cam_id: int):
    sim = Sim(drone="cf21B_500", n_drones=2)
    sim.render(mode="human", camera=cam_id)
    viewer_cam = sim.viewer.viewer.cam
    assert viewer_cam.type == mujoco.mjtCamera.mjCAMERA_FIXED, "Camera type was not set to FIXED"
    assert viewer_cam.fixedcamid == cam_id, f"Expected cam ID {cam_id}, got {viewer_cam.fixedcamid}"
    sim.close()


@pytest.mark.unit
@pytest.mark.parametrize("cam_name", ["fpv_cam:0", "track_cam:0"])
@pytest.mark.render
@skip_if_headless
def test_drone_camera_follows_drone(cam_name: str):
    sim = Sim(drone="cf21B_500", n_worlds=1, n_drones=1)
    cam_id = mujoco.mj_name2id(sim.mj_model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
    sim.render(mode="rgb_array", camera=cam_name)
    cam_pos_before = sim.mj_data.cam_xpos[cam_id].copy()
    # Teleport the drone and force a re-sync of the mjx data on the next render
    offset = np.array([1.0, 2.0, 3.0])
    states = sim.data.states.replace(pos=sim.data.states.pos + offset)
    sim.data = sim.data.replace(states=states, core=sim.data.core.replace(mjx_synced=False))
    sim.render(mode="rgb_array", camera=cam_name)
    cam_pos_after = sim.mj_data.cam_xpos[cam_id].copy()
    sim.close()
    assert np.allclose(cam_pos_after - cam_pos_before, offset, atol=1e-6), (
        f"Camera {cam_name} did not follow the drone: moved {cam_pos_after - cam_pos_before}, "
        f"expected {offset}"
    )


@pytest.mark.unit
@pytest.mark.render
@skip_if_headless
def test_render_free_camera():
    sim = Sim(drone="cf21B_500", n_drones=2)
    sim.render(mode="human")
    viewer_cam = sim.viewer.viewer.cam
    assert viewer_cam.type == mujoco.mjtCamera.mjCAMERA_FREE, "Camera type was not set to FREE"
    sim.close()
