import os
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

IMG_SIZE = 128

# Carga segura de modelos
modelo_edad = load_model("modelo/edad.keras")
modelo_emocion = load_model("modelo/emocion.keras")

def extraer_prediccion(ruta_imagen):
    """
    Procesa la imagen y retorna las predicciones de los dos modelos.
    """
    img = Image.open(ruta_imagen).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    img_array = np.expand_dims(np.array(img) / 255.0, axis=0)
    
    pred_edad = modelo_edad.predict(img_array, verbose=0)
    pred_emo = modelo_emocion.predict(img_array, verbose=0)
    
    return pred_edad, pred_emo
