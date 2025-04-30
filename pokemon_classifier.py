import os
import requests
import numpy as np
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from PIL import Image
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

# Konfiguracja
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
TARGET_SIZE = (128, 128)
HEADERS = {"User-Agent": "Mozilla/5.0"}

# Funkcja do pobierania obrazów
def download_images(pokemon_name, url):
    folder_path = os.path.join(DATASET_DIR, pokemon_name)
    
    # Sprawdź czy folder istnieje i ma wystarczająco obrazów
    if os.path.exists(folder_path) and len([f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.png', '.jpeg'))]) >= 30:
        print(f"Pomijam {pokemon_name} - wystarczająca liczba obrazów istnieje")
        return
    
    os.makedirs(folder_path, exist_ok=True)
    print(f"Pobieranie obrazów dla {pokemon_name}...")

    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        if response.status_code != 200:
            print(f"Błąd przy pobieraniu strony dla {pokemon_name}")
            return

        soup = BeautifulSoup(response.text, "html.parser")
        images_downloaded = 0
        
        for img in soup.find_all("img"):
            if images_downloaded >= 50:  # Limit na Pokémona
                break
                
            img_url = img.get("src")
            if not img_url:
                continue

            img_url = "https:" + img_url if img_url.startswith("//") else urljoin(url, img_url)
            
            # Pobieraj tylko obrazy
            if not any(img_url.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png']):
                continue
                
            filename = os.path.join(folder_path, os.path.basename(img_url))
            
            if not os.path.exists(filename):
                try:
                    img_data = requests.get(img_url, headers=HEADERS, timeout=10).content
                    with open(filename, "wb") as f:
                        f.write(img_data)
                    images_downloaded += 1
                    print(f"Pobrano: {filename}")
                except Exception as e:
                    print(f"Błąd przy pobieraniu {img_url}: {e}")

    except Exception as e:
        print(f"Błąd podczas przetwarzania {pokemon_name}: {e}")

# Funkcja do ładowania i przetwarzania danych
def load_and_preprocess_data():
    images = []
    labels = []
    
    for class_id, class_name in enumerate(POKEMON_CATEGORIES.keys()):
        class_dir = os.path.join(DATASET_DIR, class_name)
        if not os.path.isdir(class_dir):
            continue
            
        for img_name in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_name)
            try:
                img = Image.open(img_path).convert('RGB').resize(TARGET_SIZE)
                images.append(np.array(img) / 255.0)
                labels.append(class_id)
            except Exception as e:
                print(f"Błąd przy przetwarzaniu {img_path}: {e}")
    
    X_train, X_test, y_train, y_test = train_test_split(
        np.array(images), 
        tf.keras.utils.to_categorical(labels),
        test_size=0.3,
        random_state=42
    )
    return X_train, X_test, y_train, y_test

# Prosty model CNN
def build_simple_model(input_shape, num_classes):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def main():
    # 1. Utwórz folder główny
    os.makedirs(DATASET_DIR, exist_ok=True)
    
    # 2. Pobierz brakujące obrazy
    for name, url in POKEMON_CATEGORIES.items():
        download_images(name, url)
    
    # 3. Przygotuj dane
    X_train, X_test, y_train, y_test = load_and_preprocess_data()
    print(f"\nKształt danych treningowych: {X_train.shape}")
    print(f"Kształt danych testowych: {X_test.shape}")
    
    # 4. Zbuduj i trenuj model
    model = build_simple_model(X_train.shape[1:], len(POKEMON_CATEGORIES))
    model.summary()
    
    # Callbacki
    callbacks_list = [
        callbacks.EarlyStopping(patience=10, restore_best_weights=True),
        # Używamy teraz tylko ModelCheckpoint z nazwą docelową
        callbacks.ModelCheckpoint("simple_pokemon_model.keras", 
                                save_best_only=True,
                                monitor='val_accuracy',
                                mode='max')
    ]
    
    print("\nRozpoczęcie treningu...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=50,
        batch_size=32,
        callbacks=callbacks_list
    )
    
    # 5. Ewaluacja (używamy już zapisanego najlepszego modelu)
    print("\nOcena najlepszego modelu:")
    best_model = tf.keras.models.load_model("simple_pokemon_model.keras")
    best_model.evaluate(X_test, y_test)

    # NIE zapisujemy ponownie - już mamy najlepszą wersję

    model.save("xd.keras")

if __name__ == "__main__":
    main()