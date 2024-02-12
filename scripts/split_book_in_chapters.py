import os
import re

def split_chapters_into_files(file_path):
    # Define the pattern to match the chapter titles
    chapter_pattern = re.compile(r'^-+\s*\nChapter:\s*(.*?)\s*\n-+$', re.MULTILINE)
    
    # Get the directory name and base for the new file names
    directory = os.path.dirname(file_path)
    folder_name = os.path.basename(directory)
    
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    
    # Find all chapters using the pattern
    chapters = [match for match in chapter_pattern.finditer(content)]
    
    # Split the content into chapters and write each to a new file
    for i, match in enumerate(chapters):
        start_index = match.end()
        end_index = chapters[i + 1].start() if i + 1 < len(chapters) else len(content)
        
        # Extract the chapter title
        chapter_title = match.group(1).strip()
        # Replace spaces with underscores and remove trailing periods
        chapter_title = chapter_title.replace(' ', '_').rstrip('.')
        # Create a valid file name from it
        new_file_name = f"{folder_name}_Chapter_{i + 1}_{chapter_title}.txt"
        
        # Ensure the file name does not end with a period before the extension
        if new_file_name.endswith('.txt'):
            new_file_name = new_file_name
        else:
            new_file_name = new_file_name.rstrip('.') + '.txt'
        
        # Write the chapter to a new file
        with open(os.path.join(directory, new_file_name), 'w', encoding='utf-8') as new_file:
            new_file.write(content[start_index:end_index].strip())
        
        print(f"Chapter {i + 1}: '{chapter_title}' has been saved to '{new_file_name}'.")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python script.py <file_path>")
    else:
        file_path = sys.argv[1]
        split_chapters_into_files(file_path)
