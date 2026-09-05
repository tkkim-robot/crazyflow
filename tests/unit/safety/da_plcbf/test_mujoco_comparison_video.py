from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from matplotlib.figure import Figure

import crazyflow.safety.da_plcbf.mujoco_comparison_video as renderer
from crazyflow.safety.da_plcbf.mujoco_comparison_video import (
    ComparisonRenderConfig,
    ComparisonVideoTrace,
    MethodVideoTrace,
    ObstacleTrack,
    _demo_status,
    _draw_repertoire_projection,
    _execution_status,
    _has_recorded_collision,
    _has_recorded_margin_violation,
    _hud,
    _latest_coverage_probe,
    _latest_repertoire_probe,
    _probe_pause_index,
    _validate_method,
    _wind_streak_geometry,
    comparison_video_frames,
)
from crazyflow.safety.da_plcbf.online_constant_wind import (
    OnlineConstantWindResult,
    load_online_constant_wind_result,
    save_online_constant_wind_result,
)


def _method_trace(position: np.ndarray) -> MethodVideoTrace:
    steps, policies, horizon = len(position), 2, 3
    directions = np.asarray(((0.35, 0.20, 0.0), (0.35, -0.20, 0.0)))
    phase = np.linspace(0.0, 1.0, horizon)
    fallback = (
        position[:, None, None, :] + phase[None, None, :, None] * directions[None, :, None, :]
    )
    nominal = position[:, None, :] + phase[None, :, None] * np.asarray((0.45, 0.0, 0.0))
    return MethodVideoTrace(
        position=position,
        quaternion_xyzw=np.tile(np.asarray((0.0, 0.0, 0.0, 1.0)), (steps, 1)),
        nominal_rollout=nominal,
        fallback_rollouts=fallback,
        fallback_safe=np.ones((steps, policies), dtype=np.bool_),
        selected_policy=np.zeros((steps,), dtype=np.int32),
        selected_rollout=fallback[:, 0],
        intervention_world=np.tile(np.asarray((0.0, 0.1, 0.0)), (steps, 1)),
        intervention_norm=np.full((steps,), 0.1),
        descriptors=np.broadcast_to(directions, (steps, policies, 3)).copy(),
        library_version=np.arange(steps, dtype=np.int32),
        cumulative_gradient_steps=np.arange(steps, dtype=np.int32),
        diversity_loss=np.ones((steps,)),
        descriptor_target_loss=np.ones((steps,)),
        gradient_norm=np.ones((steps,)),
        parameter_update_norm=np.full((steps,), 0.01),
        minimum_library_value=np.full((steps,), 0.2),
        maximum_library_value=np.full((steps,), 0.5),
        selected_policy_value=np.full((steps,), 0.4),
        selected_policy_dual=np.full((steps,), 0.3),
        qp_valid=np.ones((steps,), dtype=np.bool_),
        used_fallback=np.zeros((steps,), dtype=np.bool_),
        degraded=np.zeros((steps,), dtype=np.bool_),
    )


@pytest.mark.unit
@pytest.mark.render
@pytest.mark.parametrize("mode", ("diagnostic", "demo"))
def test_mujoco_comparison_frame_smoke(mode: str) -> None:
    time = np.asarray((0.0, 0.1))
    position = np.asarray(((0.0, 0.0, 0.8), (0.05, 0.0, 0.8)))
    method = _method_trace(position)
    trace = ComparisonVideoTrace(
        time_seconds=time,
        goal_position=np.asarray((0.9, 0.0, 0.8)),
        obstacles=(
            ObstacleTrack(
                centers=np.tile(np.asarray((0.55, 0.0, 0.8)), (len(time), 1)),
                physical_radius=0.10,
                inflated_radius=0.18,
            ),
        ),
        true_wind=np.asarray(((0.0, 0.0, 0.0), (0.8, 0.2, 0.0))),
        estimated_wind=np.asarray(((0.0, 0.0, 0.0), (0.5, 0.1, 0.0))),
        wind_change_time=0.1,
        descriptor_targets=np.asarray(((0.35, 0.20, 0.0), (0.35, -0.20, 0.0))),
        fixed=method,
        adaptive=method,
        drone_model="cf21B_500",
        physical_model_name="cf21B_500",
    )

    frame = next(
        comparison_video_frames(
            trace,
            ComparisonRenderConfig(fps=10.0, width=960, height=540, camera_distance=2.8, mode=mode),
        )
    )

    assert frame.shape == (540, 960, 3)
    assert frame.dtype == np.uint8
    assert np.unique(frame.reshape(-1, 3), axis=0).shape[0] > 100


@pytest.mark.unit
def test_renderer_reports_certificate_max_and_actual_execution() -> None:
    method = _method_trace(np.asarray(((0.0, 0.0, 0.8), (0.05, 0.0, 0.8))))
    hud = _hud(method, 0)
    assert "library H = max +0.500" in hud
    assert "selected H +0.400" in hud
    assert "PL-CBF dual 3.00e-01" in hud
    assert "min library" not in hud
    assert _execution_status(method, 0)[0] == "QP INTERVENING"
    fallback = replace(
        method, qp_valid=np.asarray((False, True)), used_fallback=np.asarray((True, False))
    )
    assert _execution_status(fallback, 0)[0] == "EXECUTING FALLBACK"
    degraded = replace(
        method, qp_valid=np.asarray((False, True)), degraded=np.asarray((True, False))
    )
    assert _execution_status(degraded, 0)[0] == "UNCERTIFIED BEST EFFORT"
    no_certificate = replace(method, maximum_library_value=np.asarray((-0.1, 0.2)))
    assert _execution_status(no_certificate, 0)[0] == "NO COLLISION CERTIFICATE"
    legacy = replace(method, maximum_library_value=None, qp_valid=None)
    assert "library H = max unrecorded" in _hud(legacy, 0)
    assert _execution_status(legacy, 0)[0] == "COMMAND STATUS NOT RECORDED"
    assert _execution_status(replace(method, control_mode="nominal"), 0)[0] == "EXECUTING NOMINAL"
    inconsistent = replace(method, used_fallback=np.asarray((True, False)))
    with pytest.raises(ValueError, match="valid QP"):
        _validate_method(inconsistent, 2, "method")


@pytest.mark.unit
def test_same_state_probe_uses_only_measured_past_values_and_explicit_pause() -> None:
    time = np.asarray((0.0, 0.05, 0.08, 0.2))
    method = _method_trace(np.tile(np.asarray((0.0, 0.0, 0.8)), (4, 1)))
    trace = ComparisonVideoTrace(
        time_seconds=time,
        goal_position=np.asarray((1.0, 0.0, 0.8)),
        obstacles=(),
        true_wind=np.zeros((4, 3)),
        estimated_wind=np.zeros((4, 3)),
        wind_change_time=0.2,
        descriptor_targets=np.zeros((2, 3)),
        fixed=method,
        adaptive=method,
        coverage_probes={
            "time_seconds": [0.05, 0.1],
            "fixed_h": [-0.2, -0.1],
            "adaptive_h": [0.3, 0.4],
            "fixed_safe_count": [0, 0],
            "adaptive_safe_count": [1, 2],
            "source": "adaptive state and model",
        },
    )
    trace.validate()
    assert _latest_coverage_probe(trace, 0) is None
    assert _latest_coverage_probe(trace, 1) == 0
    assert _latest_coverage_probe(trace, 2) == 0
    assert _latest_coverage_probe(trace, 3) == 1
    config = ComparisonRenderConfig(probe_pause_time=0.05, probe_pause_seconds=2.0)
    config.validate()
    assert _probe_pause_index(trace, config, np.arange(4)) == 1
    with pytest.raises(ValueError, match="recorded coverage probe"):
        _probe_pause_index(trace, replace(config, probe_pause_time=0.08), np.arange(4))
    with pytest.raises(ValueError, match="sampled video frame"):
        _probe_pause_index(trace, replace(config, probe_pause_time=0.1), np.arange(4))


def _trace_for_positions(position: np.ndarray) -> ComparisonVideoTrace:
    count = len(position)
    method = _method_trace(position)
    return ComparisonVideoTrace(
        time_seconds=np.arange(count, dtype=float) * 0.1,
        goal_position=np.asarray((2.0, 0.0, 0.8)),
        obstacles=(),
        true_wind=np.zeros((count, 3)),
        estimated_wind=np.zeros((count, 3)),
        wind_change_time=(count - 1) * 0.1,
        descriptor_targets=np.zeros((2, 3)),
        fixed=method,
        adaptive=method,
    )


@pytest.mark.unit
def test_complete_fan_preserves_skill_identity_and_coincident_recorded_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trace = _trace_for_positions(np.asarray(((0.0, 0.0, 0.8), (0.05, 0.0, 0.8))))
    # A collapsed repertoire must remain collapsed, even when clearance labels differ.
    paths = np.repeat(trace.fixed.fallback_rollouts[:, :1], 2, axis=1)
    method = replace(
        trace.fixed,
        fallback_rollouts=paths,
        fallback_safe=np.asarray(((True, False), (False, True))),
    )
    lines, markers, selections = [], [], []
    monkeypatch.setattr(renderer, "_draw_goal_marker", lambda *args: None)
    monkeypatch.setattr(
        renderer,
        "_add_polyline",
        lambda sim, points, rgba, **style: lines.append((points, rgba, style)),
    )
    monkeypatch.setattr(
        renderer, "_add_endpoint_ring", lambda sim, point, rgba, **style: selections.append(point)
    )
    monkeypatch.setattr(renderer, "_add_arrow", lambda *args, **kwargs: None)
    sim = SimpleNamespace(
        viewer=SimpleNamespace(
            viewer=SimpleNamespace(add_marker=lambda **marker: markers.append(marker))
        )
    )
    renderer._draw_scene_markers(sim, trace, method, 0, ComparisonRenderConfig(mode="demo"))
    fan = lines[2:]
    assert len(fan) == 2
    for policy, (points, color, style) in enumerate(fan):
        np.testing.assert_array_equal(points, paths[0, policy])
        np.testing.assert_allclose(color, renderer._skill_color(policy, policy == 0))
        assert style["dashed"] == (policy == 1)
    endpoints = [marker["pos"] for marker in markers if np.all(marker["size"] == 0.025)]
    np.testing.assert_array_equal(endpoints, paths[0, :, -1])
    np.testing.assert_array_equal(selections, method.selected_rollout[0, -1][None])
    # Brightness and dashing encode clearance; hue keeps exactly the same policy identity.
    for policy in range(16):
        clear, blocked = renderer._skill_color(policy), renderer._skill_color(policy, False)
        np.testing.assert_allclose(clear[:3], blocked[:3] / 0.68)


@pytest.mark.unit
def test_wind_field_advects_in_world_coordinates_and_is_shared_between_panels() -> None:
    trace = _trace_for_positions(np.tile(np.asarray((0.0, 0.0, 0.8)), (4, 1)))
    trace = replace(
        trace,
        wind_change_time=0.1,
        true_wind=np.asarray(((0.0, 0.0, 0.0), (0.4, 0.2, 0.0), (0.4, 0.2, 0.0), (0.4, 0.2, 0.0))),
    )
    config = ComparisonRenderConfig(mode="demo")
    assert _wind_streak_geometry(trace, 0, config)[0].shape == (0, 3)
    p1, vector = _wind_streak_geometry(trace, 1, config)
    p2, _ = _wind_streak_geometry(trace, 2, config)
    np.testing.assert_allclose(vector, trace.true_wind[1] * config.wind_streak_exposure_seconds)
    shift = trace.true_wind[1] * 0.1
    # Culling can add/remove seeds, but every retained world-grid tracer moves by exactly v*dt.
    distances = np.linalg.norm(p1[:, None, :] + shift - p2[None, :, :], axis=-1)
    assert np.count_nonzero(np.min(distances, axis=1) < 1e-10) > 10
    shifted_method = replace(
        trace.adaptive, position=trace.adaptive.position + np.asarray((2.0, 1.0, 0.0))
    )
    pair = replace(trace, adaptive=shifted_method)
    swapped = replace(pair, fixed=pair.adaptive, adaptive=pair.fixed)
    np.testing.assert_array_equal(
        _wind_streak_geometry(pair, 2, config)[0], _wind_streak_geometry(swapped, 2, config)[0]
    )


@pytest.mark.unit
def test_repertoire_insets_use_recorded_shared_reference_and_fixed_metric_scales() -> None:
    trace = _trace_for_positions(np.tile(np.asarray((2.0, 1.0, 0.8)), (4, 1)))
    probes = {
        "time_seconds": np.asarray((0.0, 0.2)),
        "reference_position": trace.fixed.position[[0, 2]],
        "left_rollouts": trace.fixed.fallback_rollouts[[0, 2]],
        "right_rollouts": trace.adaptive.fallback_rollouts[[0, 2]],
        "left_safe": trace.fixed.fallback_safe[[0, 2]],
        "right_safe": trace.adaptive.fallback_safe[[0, 2]],
        "source": "recorded common state and common model",
    }
    trace = replace(trace, repertoire_probes=probes)
    trace.validate()
    assert _latest_repertoire_probe(trace, 1) == 0
    assert _latest_repertoire_probe(trace, 3) == 1
    figure = Figure(figsize=(16, 9))
    for side, bounds, projection in (
        ("left", (0.1, 0.1, 0.2, 0.2), (0, 1)),
        ("right", (0.5, 0.1, 0.2, 0.2), (0, 2)),
    ):
        _draw_repertoire_projection(figure, bounds, trace, side, 1, projection, 1.5, label_size=8)
    for axis in figure.axes:
        assert axis.get_xlim() == (-1.5, 1.5)
        assert axis.get_ylim() == (-1.5, 1.5)
        np.testing.assert_array_equal(
            axis.lines[0].get_xdata(), probes["left_rollouts"][1, 0, :, 0] - 2.0
        )
    invalid = replace(trace, repertoire_probes={**probes, "right_rollouts": np.zeros((2, 2, 1, 3))})
    with pytest.raises(ValueError, match="H >= 2"):
        invalid.validate()


@pytest.mark.unit
def test_margin_and_collision_history_persist_after_recovery() -> None:
    # The segment crosses the shell but passes outside the body, and finishes far away.
    trace = _trace_for_positions(np.asarray(((-1.0, 0.3, 0.8), (1.0, 0.3, 0.8), (2.0, 0.3, 0.8))))
    trace = replace(
        trace,
        obstacles=(ObstacleTrack(np.tile((0.0, 0.0, 0.8), (3, 1)), 0.1, 0.5),),
        drone_radius=0.05,
    )
    assert _has_recorded_margin_violation(trace, trace.fixed, 2)
    assert not _has_recorded_collision(trace, trace.fixed, 2)
    assert _demo_status(trace, trace.fixed, 2)[0] == "SAFETY MARGIN VIOLATED"
    through_body = replace(trace.fixed, position=trace.fixed.position * np.asarray((1.0, 0.0, 1.0)))
    assert _has_recorded_collision(trace, through_body, 2)
    assert _demo_status(trace, through_body, 2)[0] == "COLLISION RECORDED"


@pytest.mark.unit
def test_empty_collision_constraints_and_emergency_audit_fields() -> None:
    trace = _trace_for_positions(np.asarray(((0.0, 0.0, 0.8), (0.05, 0.0, 0.8))))
    method = replace(
        trace.fixed,
        maximum_library_value=np.full(2, np.inf),
        collision_constraint_active=np.zeros(2, dtype=bool),
    )
    _validate_method(method, 2, "method")
    assert renderer._demo_coverage(method, 0) == "No active collision constraint"
    with pytest.raises(ValueError, match="inactive collision"):
        _validate_method(
            replace(method, collision_constraint_active=np.ones(2, dtype=bool)), 2, "method"
        )
    emergency = replace(
        method,
        qp_valid=np.zeros(2, dtype=bool),
        used_emergency=np.asarray((True, False)),
        used_midpoint=np.asarray((False, True)),
        execution_mode=np.asarray((2, 3)),
        degraded=np.ones(2, dtype=bool),
    )
    _validate_method(emergency, 2, "method")
    assert _execution_status(emergency, 0)[0] == "EMERGENCY POLICY · UNCERTIFIED"
    assert _execution_status(emergency, 1)[0] == "MIDPOINT EMERGENCY · UNCERTIFIED"
    with pytest.raises(ValueError, match="agree with execution_mode"):
        _validate_method(replace(emergency, used_emergency=np.ones(2, dtype=bool)), 2, "method")


def _payload_trace() -> ComparisonVideoTrace:
    trace = _trace_for_positions(np.asarray(((0.0, 0.0, 0.8), (0.05, 0.0, 0.8), (0.10, 0.0, 0.8))))
    return replace(
        trace,
        drone_radius=0.05,
        payload_attachment_time_seconds=0.1,
        payload_half_extents=np.full(3, 0.025),
        payload_mass_delta_kg=0.0085,
        payload_base_mass_kg=0.034,
    )


@pytest.mark.unit
def test_prescribed_payload_box_uses_recorded_event_com_size_and_orientation() -> None:
    trace = _payload_trace()
    trace.validate()
    boxes = []
    sim = SimpleNamespace(
        viewer=SimpleNamespace(
            viewer=SimpleNamespace(add_marker=lambda **marker: boxes.append(marker))
        )
    )
    quaternion = np.tile((0.0, 0.0, np.sqrt(0.5), np.sqrt(0.5)), (3, 1))
    method = replace(trace.fixed, quaternion_xyzw=quaternion)
    renderer._draw_payload_marker(sim, trace, method, 0)
    assert not boxes
    assert "scheduled at t=0.1 s" in renderer._payload_caption(trace, 0)
    renderer._draw_payload_marker(sim, trace, method, 1)
    assert len(boxes) == 1
    np.testing.assert_array_equal(boxes[0]["pos"], method.position[1])
    np.testing.assert_array_equal(boxes[0]["size"], trace.payload_half_extents)
    np.testing.assert_allclose(
        np.asarray(boxes[0]["mat"]).reshape(3, 3),
        np.asarray(((0, -1, 0), (1, 0, 0), (0, 0, 1))),
        atol=1e-12,
    )
    assert renderer._payload_caption(trace, 1) == "+25% mass · prescribed attachment"
    renderer._draw_payload_marker(sim, trace, method, 2)
    np.testing.assert_array_equal(boxes[1]["pos"], method.position[2])
    with pytest.raises(ValueError, match="collision enclosure"):
        replace(trace, payload_half_extents=np.full(3, 0.03)).validate()
    with pytest.raises(ValueError, match="requires attachment time"):
        replace(trace, payload_base_mass_kg=None).validate()


@pytest.mark.unit
def test_payload_replay_metadata_round_trip_preserves_numerical_trajectories(
    tmp_path: Path,
) -> None:
    trace = _payload_trace()
    result = OnlineConstantWindResult(trace, {"experiment": "competent_checkpoint"})
    paths = save_online_constant_wind_result(result, tmp_path)
    restored = load_online_constant_wind_result(*paths).trace
    assert restored.payload_attachment_time_seconds == trace.payload_attachment_time_seconds
    assert restored.payload_mass_delta_kg == trace.payload_mass_delta_kg
    assert restored.payload_base_mass_kg == trace.payload_base_mass_kg
    np.testing.assert_array_equal(restored.payload_half_extents, trace.payload_half_extents)
    np.testing.assert_array_equal(restored.fixed.position, trace.fixed.position)
    np.testing.assert_array_equal(
        restored.adaptive.fallback_rollouts, trace.adaptive.fallback_rollouts
    )
    assert renderer._payload_caption(restored, 0) == renderer._payload_caption(trace, 0)
    assert renderer._payload_caption(restored, 1) == "+25% mass · prescribed attachment"


@pytest.mark.unit
def test_collision_freeze_keeps_telemetry_at_the_recorded_sample(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trace = _trace_for_positions(np.asarray(((-1.0, 0.0, 0.8), (0.0, 0.0, 0.8), (1.0, 0.0, 0.8))))
    trace = replace(trace, obstacles=(ObstacleTrack(np.tile((0.0, 0.0, 0.8), (3, 1)), 0.1, 0.2),))
    assert renderer._first_recorded_collision_index(trace, trace.fixed) == 1
    recorded_indices = []
    original = renderer._demo_execution_status

    def recorded_status(method: MethodVideoTrace, index: int) -> tuple[str, str]:
        recorded_indices.append(index)
        return original(method, index)

    monkeypatch.setattr(renderer, "_demo_execution_status", recorded_status)
    # The frozen left scene is held at the first recorded collision sample while global playback
    # continues. Its visible controller/certificate telemetry must not advance into post-impact.
    empty_scene = np.zeros((120, 120, 3), dtype=np.uint8)
    renderer._compose_demo_frame(
        trace,
        ComparisonRenderConfig(mode="demo", width=960, height=540),
        2,
        empty_scene,
        empty_scene,
        display_indices=(1, 2),
    )
    assert recorded_indices == [1, 2]


@pytest.mark.unit
def test_demo_goal_is_a_thin_horizontal_ring_with_an_open_physical_center(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lines, markers = [], []
    monkeypatch.setattr(
        renderer, "_add_polyline", lambda sim, points, rgba, **style: lines.append((points, style))
    )
    sim = SimpleNamespace(
        viewer=SimpleNamespace(viewer=SimpleNamespace(add_marker=lambda **m: markers.append(m)))
    )
    goal = np.asarray((1.7, -0.2, 0.8))
    renderer._draw_goal_marker(sim, goal, ComparisonRenderConfig(mode="demo"))
    assert not markers
    ring, style = lines[0]
    np.testing.assert_allclose(ring[:, 2], goal[2])
    np.testing.assert_allclose(np.linalg.norm(ring[:, :2] - goal[:2], axis=1), 0.14)
    np.testing.assert_allclose(ring[0], ring[-1])
    assert style["radius"] == 0.006
    assert np.min(np.linalg.norm(ring - goal, axis=1)) - style["radius"] > 0.05
    renderer._draw_goal_marker(sim, goal, ComparisonRenderConfig(mode="diagnostic"))
    assert len(markers) == 1
    np.testing.assert_array_equal(markers[0]["pos"], goal)
