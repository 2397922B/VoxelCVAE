import sys

# Check if the correct number of arguments was provided
if len(sys.argv) != 3:
    print("Usage: python extract_table_ids.py input_file.txt output_file.txt")
    sys.exit(1)

# Assign the file names from the command-line arguments
input_filename = sys.argv[1]
output_filename = sys.argv[2]

# Open the original file to read and a new file to write
with open(input_filename, 'r') as infile, open(output_filename, 'w') as outfile:
    # Iterate over each line in the file
    for line in infile:
        # Split the line into words and check if 'Table' is in the line
        parts = line.split()
        if 'Table' in parts:
            # Write the first part (the ID) to the outfile
            outfile.write(parts[0] + '\n')

print(f"All Table IDs have been written to {output_filename}.")