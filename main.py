from env_make import make_space_invaders_env


def main():
    # Generate Environment
    sp_inv_env = make_space_invaders_env(render_mode='human')
    # Reset Environment
    observation = sp_inv_env.reset()

if __name__ == "__main__":
    main()
