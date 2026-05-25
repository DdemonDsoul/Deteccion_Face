from PIL import Image
import numpy as np

IMG_SIZE = 128

def procesar_imagen(ruta_imagen):

    img = Image.open(ruta_imagen)

    img = img.convert("RGB")

    img = img.resize((IMG_SIZE, IMG_SIZE))

    img = np.array(img)

    img = img / 255.0

    return img
