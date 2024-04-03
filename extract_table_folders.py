import sys
import os
import zipfile

# Check if the correct number of arguments was provided
if len(sys.argv) != 4:
    print("Usage: python extract_folders.py zipfile.zip Table_ids.txt data_v0")
    sys.exit(1)

# Assign the file names from the command-line arguments
zip_filename = sys.argv[1]
id_filename = sys.argv[2]
target_dir = sys.argv[3]

# Create a directory to hold the extracted folders if it doesn't already exist
os.makedirs(target_dir, exist_ok=True)
print(f"Created directory: {target_dir}")

# Open the zip file
with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
    print(f"Opened {zip_filename}...")
    # Open the ID file and read IDs
    with open(id_filename, 'r') as id_file:
        print(f"Opened {id_filename}...")
        for line in id_file:
            folder_id = line.strip()
            # Extract the specific folder
            folder_to_extract = f"{target_dir}/{folder_id}/"
            print(f"Preparing to extract {folder_to_extract}...")
            for file_info in zip_ref.infolist():
                
                if file_info.filename.startswith(folder_to_extract):
                    print(f"Found {file_info.filename} for extraction...")
                    try:
                        zip_ref.extract(file_info, target_dir)
                        print(f"Successfully extracted {file_info.filename} to {target_dir}")
                    except Exception as e:
                        print(f"Error extracting {file_info.filename}: {e}")

print(f"All matching folders have been extracted to {target_dir}.")
