import os
import shutil
import sys

def main(source_dir, input_file, base_path):
    # Ensure the source directory exists
    if not os.path.isdir(source_dir):
        print(f"The source directory {source_dir} does not exist.")
        sys.exit(1)

    # Create target directories if they don't exist
    for category in ['train', 'val', 'test']:
        target_dir = os.path.join(base_path, category)
        os.makedirs(target_dir, exist_ok=True)

    # Process the input file
    with open(input_file, 'r') as file:
        for line in file:
            line = line.strip()
            if not line:
                continue  # Skip empty lines

            # Split each line into ID and category
            folder_id, category = line.split(',')

            # Construct the source and destination paths
            src_path = os.path.join(source_dir, folder_id)
            dest_path = os.path.join(base_path, category, folder_id)

            # Check if the source path exists, then copy the folder
            if os.path.exists(src_path):
                shutil.copytree(src_path, dest_path)
                print(f"Copied {src_path} to {dest_path}")

                # Remove the folder from the original directory
                shutil.rmtree(src_path) 
                print(f"Removed original folder {src_path}")
            else:
                print(f"Source path {src_path} does not exist. Skipping.")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python script.py <source_directory> <input_file> <base_path_for_directories>")
        sys.exit(1)

    source_directory = sys.argv[1]
    input_file_path = sys.argv[2]
    base_path_for_directories = sys.argv[3]  # Base path for train, val, test directories

    main(source_directory, input_file_path, base_path_for_directories)
