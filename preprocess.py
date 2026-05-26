# preproces.py
from PIL import Image
import numpy as np
from config import IMG_SIZE

def procesar_imagen(ruta_imagen):
    img = Image.open(ruta_imagen)
    img = img.convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)  # Formato final listo: (1, 128, 128, 3)
    return img
