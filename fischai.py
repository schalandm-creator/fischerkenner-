import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import os

# Titel
st.set_page_config(page_title="Fisch KI Test – EfficientNet-Lite0")
st.title("🐟 Fischarten-Erkennung Test (EfficientNet-Lite0)")
st.write("Lade ein Foto eines Fisches hoch und sieh, was das vortrainierte Modell erkennt.")

# Modell und Labels laden (nur einmal)
@st.cache_resource
def load_model_and_labels():
    model_path = "efficientnet_lite0.tflite"
    labels_path = "labels.txt"
    
    # Interpreter für TFLite
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    
    # Labels laden (ImageNet 1000 Klassen)
    with open(labels_path, "r") as f:
        labels = [line.strip() for line in f.readlines()]
    
    return interpreter, labels

interpreter, labels = load_model_and_labels()

# Bild hochladen
uploaded_file = st.file_uploader("Foto hochladen (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Hochgeladenes Bild", use_column_width=True)
    
    # Bild vorbereiten (224x224 für EfficientNet-Lite0)
    img = image.resize((224, 224))
    img_array = np.array(img, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0   # Normalisierung
    
    # Vorhersage
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    
    predictions = interpreter.get_tensor(output_details[0]['index'])[0]
    
    # Top 5 Ergebnisse
    top5_idx = np.argsort(predictions)[-5:][::-1]
    
    st.subheader("Ergebnis (Top 5)")
    for i, idx in enumerate(top5_idx):
        confidence = predictions[idx] * 100
        st.write(f"**{i+1}. {labels[idx]}** – {confidence:.2f}%")
    
    if any("fish" in labels[idx].lower() or "tench" in labels[idx].lower() or "sturgeon" in labels[idx].lower() for idx in top5_idx):
        st.success("Es wurde mindestens ein Fisch erkannt!")
    else:
        st.warning("Kein typischer Fisch erkannt – probiere ein klareres Foto.")

# Footer
st.caption("Vortrainiertes Modell: EfficientNet-Lite0 (ImageNet). Nur zum Testen. Später mit eigenen Fisch-Daten fine-tunen.")
