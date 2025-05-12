import os
import requests
import numpy as np
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from PIL import Image
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.optimizers import Adam

# Konfiguracja
POKEMON_CATEGORIES = {
    "Blastoise": "https://m.archives.bulbagarden.net/wiki/Category:Blastoise",
    "Bulbasaur": "https://m.archives.bulbagarden.net/wiki/Category:Bulbasaur",
    "Charizard": "https://m.archives.bulbagarden.net/wiki/Category:Charizard",
    "Charmander": "https://m.archives.bulbagarden.net/wiki/Category:Charmander",
    "Charmeleon": "https://m.archives.bulbagarden.net/wiki/Category:Charmeleon",
    "Ivysaur": "https://m.archives.bulbagarden.net/wiki/Category:Ivysaur",
    "Pikachu": "https://m.archives.bulbagarden.net/wiki/Category:Pikachu",
    "Squirtle": "https://m.archives.bulbagarden.net/wiki/Category:Squirtle",
    "Venusaur": "https://m.archives.bulbagarden.net/wiki/Category:Venusaur",
    "Wartortle": "https://m.archives.bulbagarden.net/wiki/Category:Wartortle",

}
DATASET_DIR = os.path.expanduser("~/Downloads/dataset")
TARGET_SIZE = (128, 128)
HEADERS = {"User-Agent": "Mozilla/5.0"}
LEARNING_RATE = 0.001  # Nowa stała dla learning rate

# Funkcja do pobierania obrazów (bez zmian)
def download_images(pokemon_name, url):
    folder_path = os.path.join(DATASET_DIR, pokemon_name)
    
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
            if images_downloaded >= 50:
                break
                
            img_url = img.get("src")
            if not img_url:
                continue

            img_url = "https:" + img_url if img_url.startswith("//") else urljoin(url, img_url)
            
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

# Funkcja do ładowania i przetwarzania danych (bez zmian)
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

# Zmodyfikowana funkcja budująca model z learning rate
def build_simple_model(input_shape, num_classes):
    # model = models.Sequential([
    #     layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
    #     layers.MaxPooling2D((3, 3)),
    #     layers.Conv2D(64, (3, 3), activation='relu'),
    #     layers.MaxPooling2D((3, 3)),
    #     layers.Flatten(),
    #     layers.Dense(128, activation='relu'),
    #     layers.Dense(num_classes, activation='softmax')
    # ])

    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.MaxPooling2D((3, 3)),

        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((3, 3)),
        layers.Dropout(0.3),

        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((3, 3)),
        layers.Dropout(0.3),

        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    # Użycie optymalizatora Adam z custom learning rate
    optimizer = Adam(learning_rate=LEARNING_RATE)
    
    model.compile(
        optimizer=optimizer,
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
        callbacks.ModelCheckpoint("simple_pokemon_model.keras", 
                                save_best_only=True,
                                monitor='val_accuracy',
                                mode='max'),
        # Dodatkowo możemy dodać redukcję learning rate
        callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5)
    ]
    
    print("\nRozpoczęcie treningu...")
    print(f"Używany learning rate: {LEARNING_RATE}")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=50,
        batch_size=16,
        callbacks=callbacks_list
    )
    
    # 5. Ewaluacja
    print("\nOcena najlepszego modelu:")
    best_model = tf.keras.models.load_model("simple_pokemon_model.keras")
    best_model.evaluate(X_test, y_test)

    model.save("xd.keras")

if __name__ == "__main__":
    main()