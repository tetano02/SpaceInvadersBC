"""
Script to merge multiple demonstration files into a single file.
Original files are kept intact.
"""

from data_manager import DataManager


def merge_demo_files():
    """
    Example usage of the merge_demonstrations method.
    You can modify this function to specify which files to merge.
    """
    # Initialize DataManager
    dm = DataManager()

    # List all available demonstration files
    print("\n📋 Available demonstration files:")
    demo_files = dm.list_demonstrations()

    if not demo_files:
        print("No demonstration files found!")
        return

    for i, demo_file in enumerate(demo_files, 1):
        info = dm.get_demonstrations_info(demo_file)
        print(f"\n{i}. {demo_file.name}")
        print(f"   Episodi: {info['num_episodes']}")
        print(f"   Steps: {info['total_steps']}")
        print(f"   Source: {info['source']}")

    # Choose files to merge
    print("\n" + "=" * 60)
    print("Enter the numbers of files to merge (separated by comma)")
    print("Example: 1,2,3 to merge the first three files")
    print("Or press ENTER to cancel")
    print("=" * 60)

    choice = input("\nFiles to merge: ").strip()

    if not choice:
        print("Operation canceled.")
        return

    # Parsing input
    try:
        indices = [int(x.strip()) for x in choice.split(",")]
        files_to_merge = [
            demo_files[i - 1] for i in indices if 1 <= i <= len(demo_files)
        ]
    except (ValueError, IndexError):
        print("❌ Invalid input!")
        return

    if not files_to_merge:
        print("❌ No files selected!")
        return

    # Confirm
    print(f"\n📦 {len(files_to_merge)} files will be merged:")
    for f in files_to_merge:
        print(f"  - {f.name}")

    confirm = input("\nConfirm? (y/n): ").strip().lower()
    if confirm != "y":
        print("Operation canceled.")
        return

    # Merge files
    merged_file, merged_id = dm.merge_demonstrations(files_to_merge)

    if merged_file:
        print(f"\n🎉 Merged file created successfully!")
        print(f"📄 Path: {merged_file}")
        print(f"🔑 ID: {merged_id}")
    else:
        print("❌ Error while merging files.")


def merge_specific_files(file_paths, output_name=None):
    """
    Merges specific files without user interaction.

    Args:
        file_paths: List of paths (str or Path) to files to merge
        output_name: Name of the output file (optional)

    Returns:
        tuple: (Path of merged file, generated ID)

    Example:
        merge_specific_files([
            'data/demonstrations/dem_251115_113635_fNAzd.pkl',
            'data/demonstrations/dem_251115_120830_nqKgz.pkl'
        ])
    """
    dm = DataManager()
    return dm.merge_demonstrations(file_paths, output_filename=output_name)


if __name__ == "__main__":
    # Interactive mode
    merge_demo_files()

    # Or use merge_specific_files directly to specify files:
    # merged_file, merged_id = merge_specific_files([
    #     'data/demonstrations/dem_251115_113635_fNAzd.pkl',
    #     'data/demonstrations/dem_251115_120830_nqKgz.pkl'
    # ])
