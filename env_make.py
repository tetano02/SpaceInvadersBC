import gymnasium as gym
import ale_py


def make_space_invaders_env(render_mode=None):
    """
    Creates and returns the Atari Space Invaders environment.

    Args:
        render_mode (str, optional): Rendering mode.
                                     Options: None, 'human', 'rgb_array'

    Returns:
        gym.Env: The configured Space Invaders environment
    """
    # Register Atari games
    gym.register_envs(ale_py)

    # Create Space Invaders environment
    env = gym.make("ALE/SpaceInvaders-v5", render_mode=render_mode)

    return env
