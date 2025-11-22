"""
Main script per il progetto di Behavioral Cloning su Space Invaders.
Permette di raccogliere dimostrazioni, addestrare il modello e valutare la policy.
"""

from pathlib import Path
from datetime import datetime
import sys


def print_banner():
    """Stampa il banner del progetto."""
    print("\n" + "=" * 60)
    print("  BEHAVIORAL CLONING - Space Invaders")
    print("  CTRL+C - CTRL+PAC")
    print("=" * 60)


def check_demonstrations():
    """Controlla se esistono dimostrazioni."""
    demo_dir = Path("data/demonstrations")
    if not demo_dir.exists():
        return False, []
    demo_files = list(demo_dir.glob("dem_*.pkl"))
    return len(demo_files) > 0, demo_files


def check_models():
    """Controlla se esistono modelli addestrati."""
    model_dir = Path("data/models")
    if not model_dir.exists():
        return False, []
    model_files = list(model_dir.glob("*.pth"))
    return len(model_files) > 0, model_files


def prompt_int_input(message: str, default: int) -> int:
    """Richiede un intero con default."""
    raw = input(message).strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        print(f"Input non valido. Uso il valore di default ({default}).")
        return default


def prompt_float_input(message: str, default: float) -> float:
    """Richiede un float con default."""
    raw = input(message).strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        print(f"Input non valido. Uso il valore di default ({default}).")
        return default


def choose_model_file(model_files):
    """Permette all'utente di scegliere un modello salvato."""
    if not model_files:
        print("\nNessun modello disponibile.")
        return None

    sorted_models = sorted(model_files, key=lambda p: p.stat().st_mtime, reverse=True)
    print("\n=== MODELLI DISPONIBILI PER GAIL ===")
    for idx, model_path in enumerate(sorted_models, start=1):
        timestamp = datetime.fromtimestamp(model_path.stat().st_mtime)
        ts_str = timestamp.strftime("%Y-%m-%d %H:%M:%S")
        print(f"  {idx}. {model_path.name} (ultimo update: {ts_str})")
    print("Inserisci il numero del modello o 'q' per annullare. [ENTER = 1]")

    while True:
        choice = input("Modello: ").strip()
        if choice.lower() in {"q", "quit", "exit"}:
            return None
        if not choice:
            return sorted_models[0]
        if choice.isdigit():
            idx = int(choice)
            if 1 <= idx <= len(sorted_models):
                return sorted_models[idx - 1]
        print("Input non valido, riprova.")


def improve_model_with_gail(demo_files, model_files):
    """Flusso interattivo per migliorare una policy tramite GAIL."""

    if not demo_files:
        print("\n⚠ Sono necessarie dimostrazioni per eseguire GAIL.")
        return

    from data_manager import DataManager
    from behavioral_cloning import (
        build_policy,
        DEFAULT_MODEL_TYPE,
        select_demonstration_files,
        select_device,
    )
    from discriminator import Discriminator
    from env_make import make_space_invaders_env
    from evaluate_bc import BCAgent
    from gail_trainer import GAILTrainer, FrameStackPreprocessor

    data_manager = DataManager()
    model_path = choose_model_file(model_files)
    if model_path is None:
        print("\nOperazione annullata: nessun modello selezionato.")
        return

    checkpoint = data_manager.load_model(model_path, device="cpu")
    run_metadata = checkpoint.get("run_metadata") or {}
    model_type = run_metadata.get("model_type", DEFAULT_MODEL_TYPE)
    frame_mode = BCAgent._determine_frame_mode(run_metadata, checkpoint) or "single"
    frame_mode = frame_mode.lower()
    stack_size = 2 if frame_mode == "stacked" else 1

    print(f"\nModello selezionato: {model_path.name}")
    print(f"Tipo architettura: {model_type} | Input: {frame_mode}")

    selected_demo_files = select_demonstration_files(demo_files)
    if not selected_demo_files:
        print("\nOperazione annullata: nessuna dimostrazione selezionata.")
        return

    preview_data = data_manager.load_demonstrations(selected_demo_files[0])
    first_episode = preview_data[0]
    first_obs = first_episode["observations"][0]
    if first_obs.ndim == 3:
        base_channels = first_obs.shape[2]
        height, width = first_obs.shape[0], first_obs.shape[1]
        input_channels = base_channels * stack_size
        observation_shape = (input_channels, height, width)
    else:
        base_dim = first_obs.shape[0]
        input_channels = base_dim * stack_size
        observation_shape = (input_channels,)

    device = select_device()
    env = make_space_invaders_env(render_mode=None)
    num_actions = env.action_space.n

    policy = build_policy(model_type=model_type, num_actions=num_actions, in_channels=input_channels)
    policy_mode = "q_values" if model_type.lower() == "dqn" else "logits"
    preprocessor = FrameStackPreprocessor(stack_size=stack_size)
    discriminator = Discriminator(observation_shape=observation_shape, num_actions=num_actions)

    print("\nConfigura training GAIL:")
    num_iterations = prompt_int_input("Iterazioni GAIL [default: 20]: ", 20)
    steps_per_collect = prompt_int_input("Passi raccolti per iterazione [default: 2048]: ", 2048)
    disc_updates = prompt_int_input("Aggiornamenti discriminatore per iterazione [default: 5]: ", 5)
    policy_updates = prompt_int_input("Aggiornamenti policy per iterazione [default: 1]: ", 1)
    epsilon = prompt_float_input("Epsilon esplorazione [default: 0.05]: ", 0.05)

    trainer = GAILTrainer(
        policy=policy,
        discriminator=discriminator,
        env=env,
        expert_files=selected_demo_files,
        bc_checkpoint_path=model_path,
        policy_output_type=policy_mode,
        state_preprocessor=preprocessor,
        device=device,
    )

    try:
        trainer.train(
            num_iterations=num_iterations,
            steps_per_collect=steps_per_collect,
            discriminator_updates=disc_updates,
            policy_updates=policy_updates,
            epsilon=epsilon,
        )
    finally:
        env.close()

    print(f"\nMetriche GAIL salvate in: {trainer.metrics_csv_path}")

    save_choice = input("\nSalvare il modello migliorato? [y/N]: ").strip().lower()
    if save_choice in {"y", "yes"}:
        model_path_saved, timestamp, model_id = trainer.save_checkpoint()
        print(f"✓ Modello GAIL salvato in: {model_path_saved}")
        if timestamp and model_id:
            print(f"  ID run: {timestamp}_{model_id}")
    else:
        print("Modello non salvato. Puoi sempre rieseguire il training GAIL in seguito.")


def main():
    """Menu principale."""
    print_banner()

    while True:
        print("\n" + "-" * 60)
        print("MENU PRINCIPALE")
        print("-" * 60)

        # Controlla stato del progetto
        has_demos, demo_files = check_demonstrations()
        has_models, model_files = check_models()

        print("\nStato del progetto:")
        if has_demos:
            print(f"  ✓ Dimostrazioni disponibili: {len(demo_files)} file(s)")
        else:
            print(f"  ✗ Nessuna dimostrazione trovata")

        if has_models:
            print(f"  ✓ Modelli addestrati: {len(model_files)} file(s)")
        else:
            print(f"  ✗ Nessun modello trovato")

        print("\nOpzioni:")
        print("  1. Raccogli dimostrazioni (gioca manualmente)")
        print("  2. Carica dataset esperto da Minari")
        print("  3. Addestra modello con Behavioral Cloning (scegli architettura)")
        print("  4. Valuta policy appresa")
        print("  5. Pipeline completa (raccolta + training + valutazione)")
        print("  6. Migliora un modello con GAIL")
        print("  0. Esci")

        choice = input("\nScelta: ").strip()

        if choice == "1":
            print("\n" + "=" * 60)
            print("RACCOLTA DIMOSTRAZIONI")
            print("=" * 60)
            from collect_demonstrations import main as collect_main

            collect_main()

        elif choice == "2":
            print("\n" + "=" * 60)
            print("CARICAMENTO DATASET MINARI")
            print("=" * 60)
            from load_minari_dataset import main as minari_main

            minari_main()

        elif choice == "3":
            if not has_demos:
                print("\n⚠ Attenzione: Non ci sono dimostrazioni disponibili!")
                print("Esegui prima l'opzione 1 o 2 per raccogliere/caricare dati.")
                input("\nPremi ENTER per continuare...")
                continue

            print("\n" + "=" * 60)
            print("TRAINING BEHAVIORAL CLONING")
            print("=" * 60)
            from behavioral_cloning import main as train_main, prompt_model_type

            selected_model = prompt_model_type()
            train_main(selected_model_type=selected_model)

        elif choice == "4":
            if not has_models:
                print("\n⚠ Attenzione: Non ci sono modelli addestrati!")
                print("Esegui prima l'opzione 3 per addestrare un modello.")
                input("\nPremi ENTER per continuare...")
                continue

            print("\n" + "=" * 60)
            print("VALUTAZIONE POLICY")
            print("=" * 60)
            from evaluate_bc import main as eval_main

            eval_main()

        elif choice == "5":
            print("\n" + "=" * 60)
            print("PIPELINE COMPLETA")
            print("=" * 60)

            # Step 1: Raccolta
            print("\n[Step 1/3] Raccolta dimostrazioni...")
            from collect_demonstrations import main as collect_main

            collect_main()

            # Controlla se sono state raccolte dimostrazioni
            has_demos, _ = check_demonstrations()
            if not has_demos:
                print("\n⚠ Nessuna dimostrazione raccolta. Pipeline interrotta.")
                input("\nPremi ENTER per continuare...")
                continue

            # Step 2: Training
            print("\n[Step 2/3] Training modello...")
            from behavioral_cloning import main as train_main, prompt_model_type

            selected_model = prompt_model_type()
            train_main(selected_model_type=selected_model)

            # Controlla se il modello è stato creato
            has_models, _ = check_models()
            if not has_models:
                print("\n⚠ Nessun modello creato. Pipeline interrotta.")
                input("\nPremi ENTER per continuare...")
                continue

            # Step 3: Valutazione
            print("\n[Step 3/3] Valutazione policy...")
            from evaluate_bc import main as eval_main

            eval_main()

            print("\n✓ Pipeline completata!")

        elif choice == "6":
            if not has_models:
                print("\n⚠ Nessun modello addestrato disponibile. Esegui prima il training BC (opzione 3).")
                input("\nPremi ENTER per continuare...")
                continue
            if not has_demos:
                print("\n⚠ Servono dimostrazioni per GAIL. Raccoglile o importale con le opzioni 1 o 2.")
                input("\nPremi ENTER per continuare...")
                continue
            improve_model_with_gail(demo_files, model_files)

        elif choice == "0":
            print("\nArrivederci!")
            sys.exit(0)

        else:
            print("\n⚠ Scelta non valida! Riprova.")
            input("\nPremi ENTER per continuare...")


if __name__ == "__main__":
    main()
