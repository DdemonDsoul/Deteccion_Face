# config.py
IMG_SIZE = 128
BATCH_SIZE = 16  # Reducido de 32 a 16 para evitar la saturación de RAM y cierres inesperados
EPOCHS_RETRAIN = 5

# Las claves representan los nombres EXACTOS de tus carpetas en el disco.
# El orden alfabético de las claves coincidirá al 100% con los índices de Keras.
MAPEO_EDADES = {
    "00-17": "Menor de 17",
    "18-20": "18 a 20 años",
    "21-30": "21 a 30 años",
    "31-40": "31 a 40 años",
    "41-50": "41 a 50 años",
    "51-60": "51 a 60 años",
    "61-99": "Mayor de 60"
}

MAPEO_EMOCIONES = {
    "anger": "Enojo",
    "disgust": "Disgusto",
    "fear": "Miedo",
    "happy": "Felicidad",
    "neutral": "Neutral",
    "sad": "Tristeza",
    "surprised": "Sorpresa"
}

EDADES_CARPETAS = sorted(list(MAPEO_EDADES.keys()))
EMOCIONES_CARPETAS = sorted(list(MAPEO_EMOCIONES.keys()))
