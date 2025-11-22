"""Utility per rivedere le dimostrazioni salvate mostrando i frame originali."""

import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np

# Pygame importa ancora pkg_resources all'avvio: silenziamo il warning finché upstream non migra
warnings.filterwarnings(
    "ignore",
    message="pkg_resources is deprecated as an API.*",
    category=UserWarning,
    module=r"pygame\.pkgdata",
)

import pygame

from data_manager import DataManager


def parse_args():
    parser = argparse.ArgumentParser(
        description="Riproduce visivamente le dimostrazioni salvate",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--file", type=str, help="File di dimostrazioni da riprodurre")
    parser.add_argument(
        "--fps", type=int, default=30, help="Frame al secondo durante la riproduzione"
    )
    parser.add_argument(
        "--scale", type=float, default=3.0, help="Fattore di scala della finestra"
    )
    parser.add_argument(
        "--episode",
        type=int,
        default=None,
        help="Numero (1-based) dell'episodio da riprodurre subito",
    )
    parser.add_argument(
        "--loop",
        action="store_true",
        help="Riproduce in loop tutti gli episodi finché non si chiude la finestra",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Mostra l'elenco dei file disponibili e termina",
    )
    return parser.parse_args()


def prompt_for_demo_file(data_manager: DataManager) -> Path | None:
    files = data_manager.list_demonstrations()
    if not files:
        print("\n⚠ Nessuna dimostrazione trovata in data/demonstrations")
        return None

    if len(files) == 1:
        print(f"\nUtilizzo file: {files[0]}")
        return files[0]

    print("\nDimostrazioni disponibili:")
    for idx, file in enumerate(files, start=1):
        info = data_manager.get_demonstrations_info(file)
        print(
            f"  [{idx}] {file.name} | episodi: {info['num_episodes']} | steps: {info['total_steps']} | source: {info['source']}"
        )

    while True:
        choice = input("Seleziona il numero del file da riprodurre: ").strip()
        if not choice.isdigit():
            print("Inserisci un numero valido")
            continue
        choice_idx = int(choice)
        if 1 <= choice_idx <= len(files):
            return files[choice_idx - 1]
        print("Indice fuori range, riprova")


def load_demonstrations(file_arg: str | None, data_manager: DataManager):
    if file_arg:
        selected = Path(file_arg)
    else:
        selected = prompt_for_demo_file(data_manager)

    if selected is None:
        return None, None

    if not selected.exists():
        print(f"\n⚠ File non trovato: {selected}")
        return None, None

    demonstrations = data_manager.load_demonstrations(selected)
    return demonstrations, selected


def create_window(first_frame: np.ndarray, scale: float):
    height, width = first_frame.shape[:2]
    window_size = (int(width * scale), int(height * scale))
    screen = pygame.display.set_mode(window_size)
    pygame.display.set_caption("Replay Dimostrazioni - Space Invaders")
    return screen, pygame.time.Clock(), window_size


def draw_frame(screen, frame, window_size):
    # Pygame si aspetta superfici trasposte rispetto a NumPy (w, h, c)
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
                screen, font, "Pausa - premi SPACE per riprendere", position=(10, 10)
            )
            pygame.display.flip()
            clock.tick(15)
            continue

        frame = observations[step]
        draw_frame(screen, frame, window_size)

        overlay_text = (
            f"Ep {episode_idx + 1}/{total_episodes}  |  Step {step + 1}/{num_steps}  |  "
            f"Azione: {int(actions[step])}  |  Reward cum.: {total_reward:.1f}"
        )
        draw_overlay(screen, font, overlay_text)

        pygame.display.flip()
        clock.tick(fps)
        step += 1

    return status


def replay_demonstrations(demonstrations, fps, scale, start_episode, loop):
    if not demonstrations:
        print("\n⚠ Il file non contiene episodi da riprodurre")
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
    args = parse_args()
    data_manager = DataManager()

    if args.list:
        files = data_manager.list_demonstrations()
        if not files:
            print("\nNessuna dimostrazione trovata.")
            return 0
        print("\nFile disponibili:")
        for file in files:
            info = data_manager.get_demonstrations_info(file)
            print(
                f"- {file.name} | episodi: {info['num_episodes']} | steps: {info['total_steps']} | source: {info['source']}"
            )
        return 0

    demonstrations, selected_file = load_demonstrations(args.file, data_manager)
    if demonstrations is None:
        return 1

    print(f"\nRiproduzione dimostrazioni da: {selected_file}")
    print(
        "Controlli: SPACE/P per pausa, RIGHT episodio successivo, LEFT precedente, R restart, ESC per uscire"
    )

    pygame.init()
    try:
        replay_demonstrations(
            demonstrations,
            fps=max(1, args.fps),
            scale=max(1.0, args.scale),
            start_episode=args.episode,
            loop=args.loop,
        )
    finally:
        pygame.quit()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
