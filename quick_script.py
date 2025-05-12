import os

# Ścieżka do folderu ~/Downloads/data
data_path = os.path.expanduser("~/Downloads/dataset")

# Pobierz tylko foldery (nazwy Pokémonów)
pokemon_folders = [name for name in os.listdir(data_path)
                   if os.path.isdir(os.path.join(data_path, name))]

# Posortuj nazwy Pokémonów alfabetycznie (opcjonalnie)
pokemon_folders = sorted(pokemon_folders)

# Tworzenie słownika POKEMON_CATEGORIES
POKEMON_CATEGORIES = {
    name: f"https://m.archives.bulbagarden.net/wiki/Category:{name}"
    for name in pokemon_folders
}

# Tworzenie słownika CLASS_NAMES
CLASS_NAMES = {
    idx: name
    for idx, name in enumerate(pokemon_folders)
}

# Wypisz wynik
print("POKEMON_CATEGORIES = {")
for name, url in POKEMON_CATEGORIES.items():
    print(f'    "{name}": "{url}",')
print("}\n")

print("CLASS_NAMES = {")
for idx, name in CLASS_NAMES.items():
    print(f'    {idx}: "{name}",')
print("}")
