import subprocess

# List of Python scripts to run in order
scripts = [
    "extract_folders.py zipfile.zip Table_ids.txt data_v0"
    "script2.py",
    "script3.py"
]

# Run each script in the order specified
for script in scripts:
    try:
        print(f"Running {script}...")
        subprocess.run(["python", script], check=True)
        print(f"{script} executed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while running {script}: {str(e)}")
        break