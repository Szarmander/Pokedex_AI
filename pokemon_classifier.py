import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Mapowanie Pokémonów na ich kategorie w archiwum Bulbagarden
POKEMON_CATEGORIES = {
    "Charmander": "https://m.archives.bulbagarden.net/wiki/Category:Charmander",
    "Charmeleon": "https://m.archives.bulbagarden.net/wiki/Category:Charmeleon",
    "Charizard": "https://m.archives.bulbagarden.net/wiki/Category:Charizard",
    "Squirtle": "https://m.archives.bulbagarden.net/wiki/Category:Squirtle",
    "Wartortle": "https://m.archives.bulbagarden.net/wiki/Category:Wartortle",
    "Blastoise": "https://m.archives.bulbagarden.net/wiki/Category:Blastoise",
    "Bulbasaur": "https://m.archives.bulbagarden.net/wiki/Category:Bulbasaur",
    "Ivysaur": "https://m.archives.bulbagarden.net/wiki/Category:Ivysaur",
    "Venusaur": "https://m.archives.bulbagarden.net/wiki/Category:Venusaur",
    "Pikachu": "https://m.archives.bulbagarden.net/wiki/Category:Pikachu"
}

DATASET_DIR = "data/pokemon_dataset"
os.makedirs(DATASET_DIR, exist_ok=True)

HEADERS = {"User-Agent": "Mozilla/5.0"}

# Funkcja do pobierania obrazów
def download_images(pokemon_name, url):
    folder_path = os.path.join(DATASET_DIR, pokemon_name)

    # Sprawdzenie, czy folder już istnieje i zawiera pliki
    if os.path.exists(folder_path) and len(os.listdir(folder_path)) > 0:
        print(f"Pomijam {pokemon_name}, ponieważ folder już istnieje.")
        return

    os.makedirs(folder_path, exist_ok=True)

    response = requests.get(url, headers=HEADERS)
    if response.status_code != 200:
        print(f"Nie udało się pobrać strony dla {pokemon_name}")
        return

    soup = BeautifulSoup(response.text, "html.parser")
    images = soup.find_all("img")

    for img in images:
        img_url = img.get("src")
        if not img_url:
            continue

        if img_url.startswith("//"):
            img_url = "https:" + img_url
        else:
            img_url = urljoin(url, img_url)

        filename = os.path.join(folder_path, img_url.split("/")[-1])

        try:
            img_data = requests.get(img_url, headers=HEADERS).content
            with open(filename, "wb") as f:
                f.write(img_data)
            print(f"Pobrano: {filename}")
        except Exception as e:
            print(f"Błąd pobierania {img_url}: {e}")

# Pobieranie obrazów dla każdego Pokemona
for name, link in POKEMON_CATEGORIES.items():
    download_images(name, link)

print("Zakończono pobieranie obrazów.")

# Funkcja do przetwarzania obrazów
def resize_images(dataset_dir, target_size=(128, 128)):
    for pokemon_name in os.listdir(dataset_dir):
        pokemon_dir = os.path.join(dataset_dir, pokemon_name)
        if os.path.isdir(pokemon_dir):
            for filename in os.listdir(pokemon_dir):
                file_path = os.path.join(pokemon_dir, filename)
                try:
                    with Image.open(file_path) as img:
                        img = img.resize(target_size)
                        img.save(file_path)
                except Exception as e:
                    print(f"Błąd przetwarzania {file_path}: {e}")

resize_images(DATASET_DIR)

# Funkcja do tworzenia datasetu
def create_dataset(dataset_dir, target_size=(128, 128)):
    images = []
    labels = []
    label_dict = {}
    current_label = 0

    for pokemon_name in os.listdir(dataset_dir):
        pokemon_dir = os.path.join(dataset_dir, pokemon_name)
        if os.path.isdir(pokemon_dir):
            label_dict[current_label] = pokemon_name
            for filename in os.listdir(pokemon_dir):
                file_path = os.path.join(pokemon_dir, filename)
                try:
                    with Image.open(file_path) as img:
                        img = img.resize(target_size)
                        
                        # Przekształć obraz do formatu RGB (3 kanały)
                        if img.mode != 'RGB':
                            img = img.convert('RGB')
                        
                        img_array = np.array(img) / 255.0  # Normalizacja
                        images.append(img_array)
                        labels.append(current_label)
                except Exception as e:
                    print(f"Błąd przetwarzania {file_path}: {e}")
            current_label += 1

    images = np.array(images)
    labels = to_categorical(np.array(labels))

    return images, labels, label_dict

images, labels, label_dict = create_dataset(DATASET_DIR)
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Funkcja do tworzenia modelu
def create_model(input_shape, num_classes):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    return model

input_shape = X_train.shape[1:]
num_classes = len(label_dict)
model = create_model(input_shape, num_classes)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Trenowanie modelu
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Ewaluacja modelu
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Loss: {loss}, Accuracy: {accuracy}")

# Zapisywanie modelu
model.save("data/pokemon_classifier.h5")

# Funkcja do przewidywania Pokémona na podstawie obrazu
def predict_pokemon(model, image_path, label_dict, target_size=(128, 128)):
    with Image.open(image_path) as img:
        img = img.resize(target_size)
        
        # Przekształć obraz do formatu RGB (3 kanały)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        prediction = model.predict(img_array)
        predicted_label = np.argmax(prediction)
        return label_dict[predicted_label]

# Przykład użycia
file_names = [f for f in os.listdir("data/test_dataset") if os.path.isfile(os.path.join("data/test_dataset", f))]
for file_name in file_names:
    pokemon_name = predict_pokemon(model, f"data/test_dataset/{file_name}", label_dict)
    print(f"Obecny plik: {file_name} | Przewidywany Pokémon to: {pokemon_name}")
