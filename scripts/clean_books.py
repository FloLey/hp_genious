import sys
import re

def remove_pipes_and_whitespace_before_punctuation(file_path):
    try:
        # Open the file and read its content
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        
        # Remove all '|' characters
        content_no_pipes = content.replace('|', '')
        
        # Remove whitespace before punctuation
        # The regex looks for one or more whitespace characters (\s+) followed by a punctuation mark
        # and replaces it with the punctuation mark alone
        modified_content = re.sub(r'\s+([,.!?;:])', r'\1', content_no_pipes)
        
        # Write the modified content back to the file
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(modified_content)
        
        print(f"All '|' characters and whitespace before punctuation have been successfully removed from '{file_path}'.")
    
    except FileNotFoundError:
        print(f"File '{file_path}' not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <file_path>")
    else:
        file_path = sys.argv[1]
        remove_pipes_and_whitespace_before_punctuation(file_path)
