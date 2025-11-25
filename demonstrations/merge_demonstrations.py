"""
Script per unire più file di dimostrazioni in un unico file.
I file originali vengono mantenuti intatti.
"""

from pathlib import Path
from data_manager import DataManager


def merge_demo_files():
    """
    Esempio di utilizzo del metodo merge_demonstrations.
    Puoi modificare questa funzione per specificare i file da unire.
    """
    # Inizializza il DataManager
    dm = DataManager()
    
    # Lista tutti i file di dimostrazioni disponibili
    print("\n📋 File di dimostrazioni disponibili:")
    demo_files = dm.list_demonstrations()
    
    if not demo_files:
        print("Nessun file di dimostrazioni trovato!")
        return
    
    for i, demo_file in enumerate(demo_files, 1):
        info = dm.get_demonstrations_info(demo_file)
        print(f"\n{i}. {demo_file.name}")
        print(f"   Episodi: {info['num_episodes']}")
        print(f"   Steps: {info['total_steps']}")
        print(f"   Source: {info['source']}")
    
    # Scegli i file da unire
    print("\n" + "="*60)
    print("Inserisci i numeri dei file da unire (separati da virgola)")
    print("Esempio: 1,2,3 per unire i primi tre file")
    print("Oppure premi INVIO per annullare")
    print("="*60)
    
    choice = input("\nFile da unire: ").strip()
    
    if not choice:
        print("Operazione annullata.")
        return
    
    # Parsing dell'input
    try:
        indices = [int(x.strip()) for x in choice.split(',')]
        files_to_merge = [demo_files[i-1] for i in indices if 1 <= i <= len(demo_files)]
    except (ValueError, IndexError):
        print("❌ Input non valido!")
        return
    
    if not files_to_merge:
        print("❌ Nessun file selezionato!")
        return
    
    # Conferma
    print(f"\n📦 Verranno uniti {len(files_to_merge)} file:")
    for f in files_to_merge:
        print(f"  - {f.name}")
    
    confirm = input("\nConfermi? (s/n): ").strip().lower()
    if confirm != 's':
        print("Operazione annullata.")
        return
    
    # Unisci i file
    merged_file, merged_id = dm.merge_demonstrations(files_to_merge)
    
    if merged_file:
        print(f"\n🎉 File unito creato con successo!")
        print(f"📄 Percorso: {merged_file}")
        print(f"🔑 ID: {merged_id}")
    else:
        print("❌ Errore durante l'unione dei file.")


def merge_specific_files(file_paths, output_name=None):
    """
    Unisce file specifici senza interazione con l'utente.
    
    Args:
        file_paths: Lista di percorsi (str o Path) ai file da unire
        output_name: Nome del file di output (opzionale)
    
    Returns:
        tuple: (Path del file unito, ID generato)
    
    Esempio:
        merge_specific_files([
            'data/demonstrations/dem_251115_113635_fNAzd.pkl',
            'data/demonstrations/dem_251115_120830_nqKgz.pkl'
        ])
    """
    dm = DataManager()
    return dm.merge_demonstrations(file_paths, output_filename=output_name)


if __name__ == '__main__':
    # Modalità interattiva
    merge_demo_files()
    
    # Oppure usa direttamente merge_specific_files per specificare i file:
    # merged_file, merged_id = merge_specific_files([
    #     'data/demonstrations/dem_251115_113635_fNAzd.pkl',
    #     'data/demonstrations/dem_251115_120830_nqKgz.pkl'
    # ])
