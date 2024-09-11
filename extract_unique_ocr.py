import os
import json
from hashlib import md5

def get_md5_hash(file_content):
    """Return MD5 hash of the given file content."""
    return md5(json.dumps(file_content, sort_keys=True).encode('utf-8')).hexdigest()

def process_json_files(folder_path):
    file_hashes = {}
    json_files = [f for f in os.listdir(folder_path) if f.endswith('.json')]
    
    for file_name in json_files:
        file_path = os.path.join(folder_path, file_name)

        with open(file_path, 'r') as json_file:
            try:
                data = json.load(json_file)
            except json.JSONDecodeError:
                print(f"Could not decode {file_name}. Skipping.")
                continue

        if not data.get('lines', []):
            os.remove(file_path) 
            print(f"Deleted {file_name} (empty 'lines' array).")
            continue

        file_hash = get_md5_hash(data)
        if file_hash in file_hashes:
            file_hashes[file_hash].append(file_path)
        else:
            file_hashes[file_hash] = [file_path]

    for file_list in file_hashes.values():
        if len(file_list) > 1:
            for duplicate_file in file_list[1:]:
                os.remove(duplicate_file)
                print(f"Deleted {os.path.basename(duplicate_file)} (duplicate of {os.path.basename(file_list[0])}).")

folder_path = './Test_Dataset/OCR/Experimental_8_2'
process_json_files(folder_path)
