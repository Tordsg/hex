import os

def extract_source_code(folder_path, output_file):
    with open(output_file, 'w') as output:
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)
                    with open(file_path, 'r') as f:
                        output.write(f'File: {file_path}\n')
                        output.write(f.read())
                        output.write('\n\n')

# Example usage:
folder_path = '/Users/tord/Code/Prosjekt_Hex/hex'  # Specify the folder containing the Python files
output_file = 'source_code.txt'  # Specify the output text file name

extract_source_code(folder_path, output_file)
print(f'Source code extracted and saved to {output_file}.')
