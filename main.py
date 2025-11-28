"""
Main script for the Behavioral Cloning project on Space Invaders.
Allows collecting demonstrations, training the model, and evaluating the policy.
"""

from pathlib import Path
from datetime import datetime
import sys


def print_banner():
    """Prints the project banner."""
    print("\n" + "=" * 60)
    print("  BEHAVIORAL CLONING - Space Invaders")
    # print("  CTRL+C - CTRL+PAC")
    print("=" * 60)


def check_demonstrations():
    """Checks if demonstrations exist."""
    demo_dir = Path("data/demonstrations")
    if not demo_dir.exists():
        return False, []
    demo_files = list(demo_dir.glob("dem_*.pkl"))
    return len(demo_files) > 0, demo_files


def check_models():
    """Checks if trained models exist."""
    model_dir = Path("data/models")
    if not model_dir.exists():
        return False, []
    model_files = list(model_dir.glob("*.pth"))
    return len(model_files) > 0, model_files


def prompt_int_input(message: str, default: int) -> int:
    """Prompts for an integer with default value."""
    raw = input(message).strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        print(f"Invalid input. Using default value ({default}).")
        return default


def prompt_float_input(message: str, default: float) -> float:
    """Prompts for a float with default value."""
    raw = input(message).strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        print(f"Invalid input. Using default value ({default}).")
        return default

def main():
    """Main menu."""

    print_banner()

    while True:
        print("\n" + "-" * 60)
        print("MAIN MENU")
        print("-" * 60)

        # check state of project
        has_demos, demo_files = check_demonstrations()
        has_models, model_files = check_models()

        print("\nState of the project:")
        if has_demos:
            print(f"  ✓ Available demonstrations: {len(demo_files)} file(s)")
        else:
            print(f"  ✗ No demonstrations found")

        if has_models:
            print(f"  ✓ Trained models: {len(model_files)} file(s)")
        else:
            print(f"  ✗ No model found")

        print("\nOptions:")
        print("  1. Collect demonstrations (play manually)")
        print("  2. Load expert dataset from Minari")
        print("  3. Train model with Behavioral Cloning (choose architecture)")
        print("  4. Test learned policy")
        print("  5. Analyze saved demonstrations")
        print("  6. Replay saved demonstrations")
        print("  7. Improve a model with GAIL")
        print("  0. Exit")

        choice = input("\nChoice: ").strip()

        if choice == "1":
            print("\n" + "=" * 60)
            print("COLLECT DEMONSTRATIONS")
            print("=" * 60)
            from demonstrations.collect_demonstrations import main as collect_main

            collect_main()

        elif choice == "2":
            print("\n" + "=" * 60)
            print("LOAD MINARI DATASET")
            print("=" * 60)
            from demonstrations.load_minari_dataset import main as minari_main

            minari_main()

        elif choice == "3":
            if not has_demos:
                print("\n⚠ Warning: No demonstrations available!")
                print("Run option 1 or 2 first to collect/load data.")
                input("\nPress ENTER to continue...")
                continue

            print("\n" + "=" * 60)
            print("BEHAVIORAL CLONING TRAINING")
            print("=" * 60)
            from behavioral_cloning import main as train_main, prompt_model_type

            selected_model = prompt_model_type()
            train_main(selected_model_type=selected_model)

        elif choice == "4":
            if not has_models:
                print("\n⚠ Warning: No trained models available!")
                print("Run option 3 first to train a model.")
                input("\nPress ENTER to continue...")
                continue

            print("\n" + "=" * 60)
            print("POLICY EVALUATION")
            print("=" * 60)
            from test_gail_model import main as test_main

            test_main()

        elif choice == "5":
            print("\n" + "=" * 60)
            print("DEMONSTRATIONS ANALYSIS")
            print("=" * 60)
            from demonstrations.analyze_demonstrations import main as analyze_main

            analyze_main()

        elif choice == "6":
            print("\n" + "=" * 60)
            print("REPLAY DEMONSTRATIONS")
            print("=" * 60)
            from demonstrations.replay_demonstrations import main as replay_main

            replay_main()

        elif choice == "7":
            if not has_models:
                print(
                    "\n⚠ No trained model available. Run BC training first (option 3)."
                )
                input("\nPress ENTER to continue...")
                continue
            if not has_demos:
                print(
                    "\n⚠ Demonstrations required for GAIL. Collect or import them with options 1 or 2."
                )
                input("\nPress ENTER to continue...")
                continue

            print("\n" + "=" * 60)
            print("GAIL TRAINING")
            print("=" * 60)
            from gail.gail_trainer import main as gail_main

            gail_main(demo_files=demo_files, model_files=model_files)

        elif choice == "0":
            print("\nGoodbye!")
            sys.exit(0)

        else:
            print("\n⚠ Invalid choice! Try again.")
            input("\nPress ENTER to continue...")


if __name__ == "__main__":
    main()
