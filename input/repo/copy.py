import os
import shutil

# List of CSV files to copy
csv_files = [
    "ada-eng.csv",
    "abr-eng.csv",
    "naw-eng.csv",
    "sil-eng.csv",
    "nko-eng.csv",
    "bim-eng.csv",
    "bwu-eng.csv",
    "akp-eng.csv",
    "bib-eng.csv",
    "biv-eng.csv",
    "gaa-eng.csv",
    "tpm-eng.csv",
    "sig-eng.csv",
    "any-eng.csv",
    "snw-eng.csv",
    "maw-eng.csv",
    "tcd-eng.csv",
]

# Create the reprocess folder if it doesn't exist
reprocess_folder = "reprocess"
os.makedirs(reprocess_folder, exist_ok=True)

# Copy each CSV file to the reprocess folder
for csv_file in csv_files:
    if os.path.exists(csv_file):
        # Get just the filename from the path
        filename = os.path.basename(csv_file)
        destination = os.path.join(reprocess_folder, filename)
        
        # Copy the file
        shutil.copy2(csv_file, destination)
        print(f"Copied: {csv_file} -> {destination}")
    else:
        print(f"File not found: {csv_file}")

print(f"\nAll files copied to '{reprocess_folder}/' folder!")
