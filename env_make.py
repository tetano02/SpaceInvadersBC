import gymnasium as gym
import ale_py


def make_space_invaders_env(render_mode=None):
    """
    Crea e restituisce l'ambiente Space Invaders di Atari.
    
    Args:
        render_mode (str, optional): Modalità di rendering. 
                                     Opzioni: None, 'human', 'rgb_array'
    
    Returns:
        gym.Env: L'ambiente Space Invaders configurato
    """
    # Registra i giochi Atari
    gym.register_envs(ale_py)
    
    # Crea l'ambiente Space Invaders
    env = gym.make('ALE/SpaceInvaders-v5', render_mode=render_mode)
    
    return env