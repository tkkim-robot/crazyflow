import jax.numpy as jnp
import numpy as np
import pytest
from jax.scipy.spatial.transform import Rotation as R

from crazyflow.control import Control
from crazyflow.dynamics import Dynamics
from crazyflow.sim import Sim
from crazyflow.sim.integration import Integrator, _integrate_symplectic, integrate_symplectic


@pytest.mark.unit
def test_symplectic_vectorization_matches_analytic_batch_and_rotor_update():
    pos = jnp.array([[[0.1, -0.2, 0.3]], [[-0.4, 0.5, 0.6]]])
    quat = jnp.broadcast_to(jnp.array([0.0, 0.0, 0.0, 1.0]), (2, 1, 4))
    vel = jnp.array([[[1.0, 2.0, -1.0]], [[-2.0, 0.5, 1.5]]])
    ang_vel = jnp.array([[[0.1, -0.2, 0.3]], [[-0.4, 0.2, 0.1]]])
    rotor_vel = jnp.arange(8.0).reshape(2, 1, 4)
    acceleration = jnp.array([[[0.5, -1.0, 2.0]], [[1.5, 0.2, -0.5]]])
    angular_acceleration = jnp.array([[[0.2, 0.3, -0.1]], [[-0.1, 0.4, 0.2]]])
    rotor_acceleration = jnp.array([[[10.0, 20.0, 30.0, 40.0]], [[-5.0, -10.0, -15.0, -20.0]]])
    dt = 0.2

    next_pos, next_quat, next_vel, next_ang_vel, next_rotor_vel = _integrate_symplectic(
        pos,
        quat,
        vel,
        ang_vel,
        rotor_vel,
        acceleration,
        angular_acceleration,
        rotor_acceleration,
        dt,
    )
    expected_vel = vel + acceleration * dt
    expected_ang_vel = ang_vel + angular_acceleration * dt
    expected_quat = (R.from_quat(quat) * R.from_rotvec(expected_ang_vel * dt)).as_quat()

    np.testing.assert_allclose(next_vel, expected_vel, rtol=1e-7, atol=1e-7)
    np.testing.assert_allclose(next_ang_vel, expected_ang_vel, rtol=1e-7, atol=1e-7)
    np.testing.assert_allclose(next_rotor_vel, rotor_vel + rotor_acceleration * dt)
    np.testing.assert_allclose(next_pos, pos + expected_vel * dt, rtol=1e-7, atol=1e-7)
    np.testing.assert_allclose(next_quat, expected_quat, rtol=1e-6, atol=1e-6)


@pytest.mark.unit
def test_symplectic_integrator_treats_dt_as_static_scalar():
    """Regression test for vectorizing over the rotor derivative instead of scalar ``dt``."""
    sim = Sim(
        n_worlds=2,
        control=Control.force_torque,
        integrator=Integrator.symplectic_euler,
        enable_mjx=False,
    )
    force_torque = jnp.zeros((sim.n_worlds, sim.n_drones, 4))
    sim.force_torque_control(force_torque)

    sim.step()
    data = sim.data

    assert data.states.pos.shape == (2, 1, 3)
    assert jnp.all(jnp.isfinite(data.states.pos))
    assert jnp.all(jnp.isfinite(data.states.rotor_vel))


@pytest.mark.unit
@pytest.mark.parametrize("dynamics", list(Dynamics))
def test_symplectic_sim_step_is_finite_for_every_dynamics_mode(dynamics: Dynamics):
    sim = Sim(
        n_worlds=2, dynamics=dynamics, integrator=Integrator.symplectic_euler, enable_mjx=False
    )

    sim.step()

    for leaf in (
        jnp.array(sim.data.states.pos),
        jnp.array(sim.data.states.quat),
        jnp.array(sim.data.states.vel),
    ):
        assert jnp.all(jnp.isfinite(leaf))


@pytest.mark.unit
def test_symplectic_wrapper_uses_force_and_torque_acceleration_not_velocity():
    sim = Sim(
        dynamics=Dynamics.first_principles,
        control=Control.rotor_vel,
        integrator=Integrator.symplectic_euler,
        freq=100,
        device="cpu",
        enable_mjx=False,
    )
    state = sim.data.states.replace(
        pos=sim.data.states.pos.at[0, 0].set(jnp.array([0.0, 0.0, 1.0])),
        vel=sim.data.states.vel.at[0, 0].set(jnp.array([0.4, -0.2, 0.1])),
        ang_vel=sim.data.states.ang_vel.at[0, 0].set(jnp.array([0.2, -0.1, 0.3])),
        rotor_vel=jnp.zeros_like(sim.data.states.rotor_vel),
    )
    data = sim.data.replace(states=state)
    derivative = data.replace(
        states_deriv=data.states_deriv.replace(
            vel=jnp.full_like(data.states_deriv.vel, 91.0),
            ang_vel=jnp.full_like(data.states_deriv.ang_vel, -73.0),
            acc=data.states_deriv.acc.at[0, 0].set(jnp.array([1.5, -2.0, 0.5])),
            ang_acc=data.states_deriv.ang_acc.at[0, 0].set(jnp.array([-0.4, 0.6, 0.2])),
            rotor_acc=jnp.zeros_like(data.states_deriv.rotor_acc),
        )
    )

    integrated = integrate_symplectic(data, derivative, dt=0.01)

    expected_velocity = np.array([0.415, -0.22, 0.105])
    expected_ang_velocity = np.array([0.196, -0.094, 0.302])
    np.testing.assert_allclose(integrated.states.vel[0, 0], expected_velocity, atol=2e-7)
    np.testing.assert_allclose(
        integrated.states.pos[0, 0], np.array([0.00415, -0.0022, 1.00105]), atol=2e-7
    )
    np.testing.assert_allclose(integrated.states.ang_vel[0, 0], expected_ang_velocity, atol=2e-7)
