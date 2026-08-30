import numpy as np
import pytest

from crazyflow.control import Control
from crazyflow.drones import load_params
from crazyflow.dynamics import Dynamics
from crazyflow.envs.drone_env import DroneEnv, action_space
from crazyflow.envs.landing_env import LandingEnv
from crazyflow.envs.reach_pos_env import ReachPosEnv
from crazyflow.envs.reach_vel_env import ReachVelEnv


@pytest.mark.unit
def test_attitude_action_space_uses_rpyt_order():
    """The first three entries are RPY angles and collective thrust is last."""
    drone = "cf2x_L250"
    space = action_space(Control.attitude, drone)
    params = load_params(drone)

    assert space.shape == (4,)
    np.testing.assert_allclose(space.low[:3], -np.pi / 2)
    np.testing.assert_allclose(space.high[:3], np.pi / 2)
    assert space.low[3] == pytest.approx(params["thrust_min"] * 4)
    assert space.high[3] == pytest.approx(params["thrust_max"] * 4)


@pytest.mark.unit
def test_force_torque_action_space_has_four_components():
    """Force/torque commands are [collective force, tx, ty, tz]."""
    space = action_space(Control.force_torque, "cf2x_L250")

    assert space.shape == (4,)
    np.testing.assert_array_equal(space.low, -np.ones(4))
    np.testing.assert_array_equal(space.high, np.ones(4))


@pytest.mark.unit
@pytest.mark.parametrize("env_type", (ReachPosEnv, ReachVelEnv, LandingEnv))
def test_builtin_environments_accept_documented_drone_argument(env_type: type[DroneEnv]):
    """All built-in environment constructors must forward the documented drone selection."""
    env = env_type(num_envs=1, drone="cf21B_500")

    assert env.sim.drone == "cf21B_500"
    env.close()


@pytest.mark.unit
def test_existing_positional_environment_arguments_remain_compatible():
    """Adding drone selection must not reinterpret the established positional arguments."""
    landing = LandingEnv(1, 10.0, Dynamics.so_rpy, 250, "cpu")
    reach_pos = ReachPosEnv(None, None, -1.0, 1.0, 1, 10.0, Dynamics.so_rpy, 250, "cpu")

    assert landing.freq == reach_pos.freq == 250
    assert landing.sim.drone == reach_pos.sim.drone == "cf2x_L250"
    landing.close()
    reach_pos.close()
