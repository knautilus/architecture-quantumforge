import os
import re
import csv
import json

IN_DIR = "original"
OUT_DIR = "processed"

def load_terms_map(path):
    with open(path, "r", encoding="utf-8") as f:
        m = json.load(f)
    return m

# Defining a function for replacement
def replace_func(match, mapping):
    return mapping[match.group(0)]  # Returning the replacement from the dictionary

def process_folder(input_folder, output_folder):
    processed_count = 0
    mapping = load_terms_map("terms_map.json")
    for filename in os.listdir(input_folder):
        old_filepath = os.path.join(input_folder, filename)
        if os.path.isfile(old_filepath):
            try:
                with open(old_filepath, 'r', encoding='utf-8') as f:
                    file_content = f.read()
            except Exception as e:
                print(f"Ошибка при чтении файла '{filename}': {e}.")
                continue

            pattern = re.compile("|".join(re.escape(k) for k in mapping))  # Creating a pattern for all keys

            replaced_content = pattern.sub(lambda m: replace_func(m, mapping), file_content)  # Replacing using the pattern
            new_filename = pattern.sub(lambda m: replace_func(m, mapping), filename)  # Replacing using the pattern
            new_filepath = os.path.join(output_folder, new_filename)
            
            try:
                with open(new_filepath, 'w', encoding='utf-8') as f:
                    f.write(replaced_content)
                print(f"Файл '{filename}' обработан, сохранён как '{new_filename}'.")
                processed_count += 1
            except Exception as e:
                print(f"Ошибка при записи файла '{new_filepath}': {e}.")
    print(f"\nОбработка завершена. Всего обработано файлов: {processed_count}.")

if __name__ == "__main__":
    process_folder(IN_DIR, OUT_DIR)