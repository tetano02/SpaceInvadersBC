"""
Main script per il progetto di Behavioral Cloning su Space Invaders.
Permette di raccogliere dimostrazioni, addestrare il modello e valutare la policy.
"""

from pathlib import Path
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
        print("  4. Test policy appresa")
        print("  5. Analizza dimostrazioni salvate")
        print("  6. Riproduci dimostrazioni salvate")
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
            print("ANALISI DIMOSTRAZIONI")
            print("=" * 60)
            from analyze_demonstrations import main as analyze_main

            analyze_main()
        
        elif choice == "6":
            print("\n" + "=" * 60)
            print("RIPRODUZIONE DIMOSTRAZIONI")
            print("=" * 60)
            from replay_demonstrations import main as reply_main

            reply_main()

        elif choice == "0":
            print("\nArrivederci!")
            sys.exit(0)

        else:
            print("\n⚠ Scelta non valida! Riprova.")
            input("\nPremi ENTER per continuare...")


if __name__ == "__main__":
    main()
