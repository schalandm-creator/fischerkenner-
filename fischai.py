import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import os

st.set_page_config(page_title="Fisch KI Test", layout="centered")
st.title("🐟 Fischarten-Erkennung Test (EfficientNet-Lite0)")
st.write("Lade ein Fisch-Foto hoch und teste das vortrainierte Modell.")

@st.cache_resource
def load_model_and_labels():
    model_path = "efficientnet_lite0.tflite"
    labels_path = "labels.txt"
    
    if not os.path.exists(model_path):
        st.error(f"Modell {model_path} nicht gefunden!")
        st.stop()
    if not os.path.exists(labels_path):
        st.error(f"Labels {labels_path} nicht gefunden!")
        st.stop()
    
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    
    with open(labels_path, "r", encoding="utf-8") as f:
        labels = [line.strip() for line in f.readlines()]
    
    return interpreter, labels

try:
    interpreter, labels = load_model_and_labels()
except Exception as e:
    st.error(f"Fehler beim Laden des Modells: {e}")
    st.stop()

uploaded_file = st.file_uploader("Foto hochladen (JPG, JPEG, PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Dein hochgeladenes Bild", use_column_width=True)
        
        # Vorbereitung für EfficientNet-Lite0 (224x224)
        img = image.resize((224, 224))
        img_array = np.array(img, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # Vorhersage
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        interpreter.set_tensor(input_details[0]['index'], img_array)
        interpreter.invoke()
        
        predictions = interpreter.get_tensor(output_details[0]['index'])[0]
        
        # Top 5
        top5_idx = np.argsort(predictions)[-5:][::-1]
        
        st.subheader("🔍 Top 5 Vorhersagen")
        for i, idx in enumerate(top5_idx):
            confidence = float(predictions[idx]) * 100
            st.write(f"**{i+1}. {labels[idx]}** — {confidence:.2f}%")
        
        # Kleiner Hinweis für Fische
        fish_keywords = ["fish", "tench", "sturgeon", "barracouta", "coho", "eel", "shark"]
        if any(any(kw in labels[idx].lower() for kw in fish_keywords) for idx in top5_idx):
            st.success("✅ Es wurde ein fischähnliches Objekt erkannt!")
        else:
            st.info("Das Modell ist noch nicht auf deutsche Fischarten trainiert – es erkennt nur allgemeine ImageNet-Klassen.")
            
    except Exception as e:
        st.error(f"Fehler bei der Verarbeitung: {e}")

st.caption("Vortrainiertes Modell: EfficientNet-Lite0 (ImageNet). Nur zum schnellen Test. Später mit eigenen Fisch-Daten fine-tunen.")
