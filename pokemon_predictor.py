import os
import numpy as np
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt

# Konfiguracja
# MODEL_PATH = "simple_pokemon_model.keras"
MODEL_PATH = "xd.keras"
TEST_DIR = "data/test_dataset"
TARGET_SIZE = (128, 128)
CLASS_NAMES = {
    0: "Charmander",
    1: "Charmeleon",
    2: "Charizard",
    3: "Squirtle",
    4: "Wartortle",
    5: "Blastoise",
    6: "Bulbasaur",
    7: "Ivysaur",
    8: "Venusaur",
    9: "Pikachu",
}

def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

def predict_images(model):
    images = []
    predictions = []
    confidences = []
    filenames = []
    
    for filename in sorted(os.listdir(TEST_DIR)):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            try:
                img_path = os.path.join(TEST_DIR, filename)
                img = Image.open(img_path).convert('RGB').resize(TARGET_SIZE)
                img_array = np.array(img) / 255.0
                
                pred = model.predict(np.expand_dims(img_array, axis=0))
                pred_class = np.argmax(pred[0])
                
                images.append(img)
                predictions.append(CLASS_NAMES[pred_class])
                confidences.append(np.max(pred[0]))
                filenames.append(filename)
                
            except Exception as e:
                print(f"Błąd przy {filename}: {str(e)}")
    
    return images, predictions, confidences, filenames

def plot_results(images, preds, confs, filenames):
    plt.figure(figsize=(15, 10))
    for i in range(len(images)):
        plt.subplot(4, 4, i+1)
        plt.imshow(images[i])
        plt.title(f"{preds[i]}\n({confs[i]:.1%})", fontsize=9)
        plt.xlabel(filenames[i], fontsize=7)
        plt.xticks([])
        plt.yticks([])
    plt.tight_layout()
    plt.show()

def main():
    model = load_model()
    images, preds, confs, filenames = predict_images(model)
    
    print("\nWyniki predykcji:")
    for f, p, c in zip(filenames, preds, confs):
        print(f"{f}: {p} ({c:.2%})")
    
    plot_results(images, preds, confs, filenames)

if __name__ == "__main__":
    main()