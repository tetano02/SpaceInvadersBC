"""
Script to collect human demonstrations by manually playing Space Invaders.
Use arrow keys to move and SPACE to fire.
"""

import gymnasium as gym
import numpy as np
from pathlib import Path
import warnings

# pygame still imports pkg_resources internally; silence its deprecation warning until upstream fixes it
warnings.filterwarnings(
    "ignore",
    message="pkg_resources is deprecated as an API.*",
    category=UserWarning,
    module=r"pygame\.pkgdata",
)

import pygame
from env_make import make_space_invaders_env
from data_manager import DataManager
from demonstrations.analyze_demonstrations import analyze_demonstrations_file, ACTION_NAMES


class DemonstrationCollector:
    """Collects human demonstrations for Behavioral Cloning."""

    def __init__(self, env, fps=30, frame_skip=2):
        self.env = env
        self.demonstrations = []
        self.current_episode = {
            "observations": [],
            "actions": [],
            "rewards": [],
            "dones": [],
        }
        
        # Clock to control frame rate
        self.clock = pygame.time.Clock()
        self.fps = fps
        self.frame_skip = frame_skip  # How many frames to execute for each recorded action

        # Key mapping -> Space Invaders actions
        # Actions: 0=NOOP, 1=FIRE, 2=RIGHT, 3=LEFT, 4=RIGHTFIRE, 5=LEFTFIRE
        self.key_to_action = {
            pygame.K_SPACE: 1,  # FIRE
            pygame.K_RIGHT: 2,  # RIGHT
            pygame.K_LEFT: 3,  # LEFT
        }

    def get_human_action(self):
        """Gets the action from keyboard input."""
        keys = pygame.key.get_pressed()

        # Key combinations
        if keys[pygame.K_RIGHT] and keys[pygame.K_SPACE]:
            return 4  # RIGHTFIRE
        elif keys[pygame.K_LEFT] and keys[pygame.K_SPACE]:
            return 5  # LEFTFIRE
        elif keys[pygame.K_RIGHT]:
            return 2  # RIGHT
        elif keys[pygame.K_LEFT]:
            return 3  # LEFT
        elif keys[pygame.K_SPACE]:
            return 1  # FIRE
        else:
            return 0  # NOOP

    def collect_episode(self):
        """Collects a single game."""
        observation, info = self.env.reset()
        done = False
        truncated = False
        total_reward = 0

        self.current_episode = {
            "observations": [],
            "actions": [],
            "rewards": [],
            "dones": [],
        }

        print("\n=== NEW GAME ===")
        print("Controls:")
        print("  ← →  : Move")
        print("  SPACE : Fire")
        print("  ESC : End collection")
        print("  Q : Quit game (don't save)")
        print("=====================\n")

        while not (done or truncated):
            # Process Pygame events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return False, total_reward
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        return False, total_reward
                    if event.key == pygame.K_q:
                        print("Game abandoned (not saved)")
                        return True, total_reward

            # Get human action
            action = self.get_human_action()

            # Save observation and action (copy necessary because gym reuses the buffer)
            self.current_episode["observations"].append(observation.copy())
            self.current_episode["actions"].append(action)

            # Execute the action for frame_skip frames and accumulate reward
            frame_reward = 0
            for _ in range(self.frame_skip):
                observation, reward, done, truncated, info = self.env.step(action)
                frame_reward += reward
                total_reward += reward
                
                if done or truncated:
                    break
                
            # Save accumulated reward and done
            self.current_episode["rewards"].append(frame_reward)
            self.current_episode["dones"].append(done or truncated)
            
            # Limit frame rate (once per action, not for each skipped frame)
            self.clock.tick(self.fps)

        # Convert lists to numpy arrays for efficiency (only once at the end)
        episode_data = {
            "observations": np.array(
                self.current_episode["observations"], dtype=np.uint8
            ),
            "actions": np.array(
                self.current_episode["actions"], dtype=np.int8
            ),
            "rewards": np.array(
                self.current_episode["rewards"], dtype=np.float32
            ),
            "dones": np.array(
                self.current_episode["dones"], dtype=np.bool_
            ),
        }

        # Save the completed episode
        self.demonstrations.append(episode_data)
        
        # Free memory of temporary lists
        self.current_episode = {
            "observations": [],
            "actions": [],
            "rewards": [],
            "dones": [],
        }
        
        print(f"\nGame completed! Total reward: {total_reward}")
        print(f"Recorded steps: {len(episode_data['actions'])}")

        return True, total_reward

    def collect_multiple_episodes(self, num_episodes=5):
        """Collects multiple games."""
        print(f"\n{'='*50}")
        print(f"DEMONSTRATIONS COLLECTION - {num_episodes} games")
        print(f"{'='*50}")

        episode_rewards = []

        for i in range(num_episodes):
            print(f"\nGame {i+1}/{num_episodes}")
            continue_collection, reward = self.collect_episode()

            if not continue_collection:
                print("\nCollection interrupted by user.")
                break

            episode_rewards.append(reward)

            if i < num_episodes - 1:
                print("\nPress a key for the next game...")
                waiting = True
                while waiting:
                    for event in pygame.event.get():
                        if event.type == pygame.KEYDOWN or event.type == pygame.QUIT:
                            waiting = False
                            break

        print(f"\n{'='*50}")
        print("COLLECTION SUMMARY")
        print(f"{'='*50}")
        print(f"Completed games: {len(self.demonstrations)}")
        if episode_rewards:
            print(f"Average reward: {np.mean(episode_rewards):.2f}")
            print(
                f"Min/max reward: {np.min(episode_rewards):.2f} / {np.max(episode_rewards):.2f}"
            )
            total_steps = sum(len(ep["actions"]) for ep in self.demonstrations)
            print(f"Total recorded steps: {total_steps}")

        return self.demonstrations

    def save_demonstrations(self, filename=None):
        """Saves demonstrations to file using DataManager."""
        data_manager = DataManager()
        return data_manager.save_demonstrations(
            demonstrations=self.demonstrations, filename=filename, source="manual"
        )


def main():
    """Main function to collect demonstrations."""
    # Request number of games BEFORE initializing pygame
    try:
        num_episodes = int(input("How many games do you want to play? [default: 1]: ") or "1")
    except ValueError:
        num_episodes = 1

    print(f"\nInitializing environment...")

    # Initialize Pygame to capture input
    pygame.init()

    # Create environment
    env = make_space_invaders_env(render_mode="human")

    # Create collector
    collector = DemonstrationCollector(env)

    # Collect demonstrations
    demonstrations = collector.collect_multiple_episodes(num_episodes)

    # Close environment and pygame BEFORE analysiss
    env.close()
    pygame.quit()

    # If there are demonstrations, analyze them and ask for confirmation
    if demonstrations:
        print("\n" + "="*60)
        print("COLLECTED DEMONSTRATIONS ANALYSIS")
        print("="*60)
        
        # Temporarily create a file for analysis
        data_manager = DataManager()
        temp_path, _ = data_manager.save_demonstrations(
            demonstrations=demonstrations,
            filename="temp_analysis.pkl",
            source="manual"
        )
        
        # Analyze
        try:
            analyze_demonstrations_file(
                temp_path,
                summary_only=False,
                data_manager=data_manager
            )
        finally:
            # Remove temporary file
            if temp_path.exists():
                temp_path.unlink()
        
        # Ask for confirmation to save
        print("\n" + "="*60)
        while True:
            choice = input("\nDo you want to save these demonstrations? (y/n): ").strip().lower()
            if choice in ['s', 'si', 'sì', 'y', 'yes']:
                saved_path, info = collector.save_demonstrations()
                print(f"\n✓ Demonstrations saved in: {saved_path}")
                break
            elif choice in ['n', 'no']:
                print("\n✗ Demonstrations discarded (not saved)")
                break
            else:
                print("Please answer 'y' for yes or 'n' for no")
    else:
        print("\nNo demonstrations to save.")

    print("\nCollection completed!")

if __name__ == "__main__":
    main()
