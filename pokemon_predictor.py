import os
import numpy as np
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt

# Konfiguracja
MODEL_PATH = "simple_pokemon_model.keras"
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
    top3_preds = []
    
    for filename in sorted(os.listdir(TEST_DIR)):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            try:
                img_path = os.path.join(TEST_DIR, filename)
                img = Image.open(img_path).convert('RGB').resize(TARGET_SIZE)
                img_array = np.array(img) / 255.0
                
                pred = model.predict(np.expand_dims(img_array, axis=0), verbose=0)
                pred = pred[0]
                pred_class = np.argmax(pred)
                top3_indices = np.argsort(pred)[-4:][::-1]  # top 4 (najlepszy + 3 kolejne)
                
                top3 = []
                for i in top3_indices[1:]:  # pomijamy pierwszy (główna predykcja)
                    top3.append((CLASS_NAMES[i], pred[i]))
                
                images.append(img)
                predictions.append(CLASS_NAMES[pred_class])
                confidences.append(pred[pred_class])
                filenames.append(filename)
                top3_preds.append(top3)
                
            except Exception as e:
                print(f"Błąd przy {filename}: {str(e)}")
    
    return images, predictions, confidences, filenames, top3_preds

def plot_results(images, preds, confs, filenames, top3_preds):
    plt.figure(figsize=(15, 10))
    for i in range(len(images)):
        plt.subplot(4, 4, i + 1)
        plt.imshow(images[i])
        title_main = f"{preds[i]} ({confs[i]:.1%})"
        top3_info = "\n".join([f"{name} ({conf:.1%})" for name, conf in top3_preds[i]])
        full_title = f"{title_main}\n{top3_info}"
        plt.title(full_title, fontsize=9)
        plt.xlabel(filenames[i], fontsize=7)
        plt.xticks([])
        plt.yticks([])
    plt.tight_layout()
    plt.show()

def main():
    model = load_model()
    images, preds, confs, filenames, top3_preds = predict_images(model)
    
    print("\nWyniki predykcji:")
    for f, p, c, top3 in zip(filenames, preds, confs, top3_preds):
        print(f"{f}: {p} ({c:.2%})")
        for alt_name, alt_conf in top3:
            print(f"   - {alt_name}: {alt_conf:.2%}")
    
    plot_results(images, preds, confs, filenames, top3_preds)

if __name__ == "__main__":
    main()
