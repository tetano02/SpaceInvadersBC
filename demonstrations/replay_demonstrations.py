"""Utility to review saved demonstrations by displaying the original frames."""

import warnings
from pathlib import Path

import numpy as np

# Pygame still imports pkg_resources at startup: silence the warning until upstream migrates
warnings.filterwarnings(
    "ignore",
    message="pkg_resources is deprecated as an API.*",
    category=UserWarning,
    module=r"pygame\.pkgdata",
)

import pygame

from data_manager import DataManager


def prompt_for_demo_file(data_manager: DataManager) -> Path | None:
    files = data_manager.list_demonstrations()
    if not files:
        print("\n⚠ No demonstrations found in data/demonstrations")
        return None

    if len(files) == 1:
        print(f"\nUsing file: {files[0]}")
        return files[0]

    print("\nAvailable demonstrations:")
    for idx, file in enumerate(files, start=1):
        info = data_manager.get_demonstrations_info(file)
        print(
            f"  [{idx}] {file.name} | episodi: {info['num_episodes']} | steps: {info['total_steps']} | source: {info['source']}"
        )

    while True:
        choice = input("Select the file number to play back: ").strip()
        if not choice.isdigit():
            print("Enter a valid number")
            continue
        choice_idx = int(choice)
        if 1 <= choice_idx <= len(files):
            return files[choice_idx - 1]
        print("Index out of range, try again")


def prompt_for_settings():
    """Prompt user for playback settings."""
    print("\n--- Playback Settings ---")

    # FPS
    while True:
        fps_input = input("FPS (frames per second, default 30): ").strip()
        if not fps_input:
            fps = 30
            break
        if fps_input.isdigit() and int(fps_input) > 0:
            fps = int(fps_input)
            break
        print("Enter a valid positive number")

    # Scale
    while True:
        scale_input = input("Window scale factor (default 3.0): ").strip()
        if not scale_input:
            scale = 3.0
            break
        try:
            scale = float(scale_input)
            if scale > 0:
                break
            print("Enter a positive number")
        except ValueError:
            print("Enter a valid number")

    # Start episode
    start_episode_input = input(
        "Start from episode number (press Enter to start from 1): "
    ).strip()
    if start_episode_input.isdigit():
        start_episode = int(start_episode_input)
    else:
        start_episode = None

    # Loop
    loop_input = input("Loop all episodes? (y/n, default n): ").strip().lower()
    loop = loop_input == "y" or loop_input == "yes"

    return fps, scale, start_episode, loop


def load_demonstrations(data_manager: DataManager):
    selected = prompt_for_demo_file(data_manager)

    if selected is None:
        return None, None

    if not selected.exists():
        print(f"\n⚠ File not found: {selected}")
        return None, None

    demonstrations = data_manager.load_demonstrations(selected)
    return demonstrations, selected


def create_window(first_frame: np.ndarray, scale: float):
    height, width = first_frame.shape[:2]
    window_size = (int(width * scale), int(height * scale))
    screen = pygame.display.set_mode(window_size)
    pygame.display.set_caption("Demonstrations Replay - Space Invaders")
    return screen, pygame.time.Clock(), window_size


def draw_frame(screen, frame, window_size):
    # Pygame expects transposed surfaces compared to NumPy (w, h, c)
    surface = pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
    if surface.get_size() != window_size:
        surface = pygame.transform.scale(surface, window_size)
    screen.blit(surface, (0, 0))


def draw_overlay(screen, font, text, position=(10, 10)):
    overlay = font.render(text, True, (255, 255, 255))
    outline = font.render(text, True, (0, 0, 0))
    screen.blit(outline, (position[0] + 1, position[1] + 1))
    screen.blit(overlay, position)


def play_episode(
    episode, screen, clock, window_size, font, fps, episode_idx, total_episodes
):
    observations = episode["observations"]
    actions = episode["actions"]
    rewards = episode["rewards"]
    total_reward = float(np.sum(rewards))
    # Use the minimum length to avoid index errors (actions may be shorter)
    num_steps = min(len(observations), len(actions))

    step = 0
    paused = False
    status = "continue"

    while step < num_steps:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return "quit"
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return "quit"
                if event.key == pygame.K_SPACE:
                    paused = not paused
                if event.key == pygame.K_RIGHT:
                    return "next"
                if event.key == pygame.K_LEFT:
                    return "previous"
                if event.key == pygame.K_r:
                    step = 0
                if event.key == pygame.K_p:
                    paused = not paused

        if paused:
            draw_overlay(
                screen, font, "Paused - press SPACE to resume", position=(10, 10)
            )
            pygame.display.flip()
            clock.tick(15)
            continue

        frame = observations[step]
        draw_frame(screen, frame, window_size)

        overlay_text = (
            f"Ep {episode_idx + 1}/{total_episodes}  |  Step {step + 1}/{num_steps}  |  "
            f"Action: {int(actions[step])}  |  Cumul. Reward: {total_reward:.1f}"
        )
        draw_overlay(screen, font, overlay_text)

        pygame.display.flip()
        clock.tick(fps)
        step += 1

    return status


def replay_demonstrations(demonstrations, fps, scale, start_episode, loop):
    if not demonstrations:
        print("\n⚠ The file contains no episodes to play back")
        return

    first_frame = demonstrations[0]["observations"][0]

    pygame.display.init()
    screen, clock, window_size = create_window(first_frame, scale)
    font = pygame.font.SysFont("Consolas", 18)

    episode_idx = max(
        0, min(len(demonstrations) - 1, (start_episode - 1) if start_episode else 0)
    )

    running = True
    while running and demonstrations:
        episode = demonstrations[episode_idx]
        status = play_episode(
            episode,
            screen,
            clock,
            window_size,
            font,
            fps,
            episode_idx,
            len(demonstrations),
        )

        if status == "quit":
            running = False
        elif status == "previous":
            episode_idx = (episode_idx - 1) % len(demonstrations)
        else:  # continue / next
            episode_idx += 1
            if episode_idx >= len(demonstrations):
                if loop:
                    episode_idx = 0
                else:
                    running = False

    pygame.display.quit()


def main():
    data_manager = DataManager()

    demonstrations, selected_file = load_demonstrations(data_manager)
    if demonstrations is None:
        return 1

    fps, scale, start_episode, loop = prompt_for_settings()

    print(f"\nPlaying back demonstrations from: {selected_file}")
    print(
        "Controls: SPACE/P for pause, RIGHT next episode, LEFT previous, R restart, ESC to exit"
    )

    pygame.init()
    try:
        replay_demonstrations(
            demonstrations,
            fps=fps,
            scale=scale,
            start_episode=start_episode,
            loop=loop,
        )
    finally:
        pygame.quit()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
