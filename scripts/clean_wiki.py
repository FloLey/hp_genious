import os
import re
import sys

# Check if a folder path is given as a command-line argument
if len(sys.argv) < 2:
    print("Please provide the folder path as a command-line argument.")
    sys.exit(1)

# Use the provided folder path
folder_path = sys.argv[1]

# Compile the regular expressions for efficiency
pattern_number_brackets = re.compile(r'\[\d+\]')
pattern_advertisement = re.compile(r'ADVERTISEMENT')
pattern_src = re.compile(r'\[src\]')

# Counter for processed files
processed_files = 0

# Iterate over each file in the folder
for filename in os.listdir(folder_path):
    # Construct the full file path
    file_path = os.path.join(folder_path, filename)
    # Check if it's a file
    if os.path.isfile(file_path):
        # Open the file, read its contents, then close it
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        
        # Perform the replacements
        content = pattern_number_brackets.sub('', content)
        content = pattern_advertisement.sub('', content)
        content = pattern_src.sub('', content)
        
        # Write the cleaned content back to the file
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(content)
        
        # Increment the processed files counter and show progress
        processed_files += 1
        print(f"Processed {filename}")

# Check if any files were processed
if processed_files == 0:
    print("No files were processed. Please check the folder path and try again.")
else:
    print(f"All files have been cleaned. Total files processed: {processed_files}")

