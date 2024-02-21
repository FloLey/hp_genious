import os
import argparse


def rename_files(root_dir):
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if '_' in filename:
                old_file_path = os.path.join(dirpath, filename)
                new_filename = filename.replace('_', '-')
                new_file_path = os.path.join(dirpath, new_filename)
                os.rename(old_file_path, new_file_path)
                print(f"Renamed '{old_file_path}' to '{new_file_path}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rename files by replacing underscores with hyphens.")
    parser.add_argument("directory", type=str, help="The root directory containing files to rename.")

    args = parser.parse_args()
    rename_files(args.directory)