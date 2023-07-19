import os
import time

def process_file(file_path):
    # Implement your processing logic here
    print(f"Processing file: {file_path}")

def check_for_new_files(directory_path):
    processed_files = set(os.listdir(directory_path))

    while True:
        files = set(os.listdir(directory_path))
        new_files = files - processed_files

        for file in new_files:
            file_path = os.path.join(directory_path, file)
            process_file(file_path)

        processed_files = files

        time.sleep(5)  # Wait for 5 seconds before the next check

if __name__ == "__main__":
    target_directory = "/home/nafaa/Desktop/stage dete"
    check_for_new_files(target_directory)

 

