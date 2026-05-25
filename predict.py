from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import os
import shutil

IMG_SIZE = 128

# ======================
# CARGAR MODELOS
# ======================

modelo_edad = load_model(
    "modelo/edad.keras"
)

modelo_emocion = load_model(
    "modelo/emocion.keras"
)

# ======================
# CLASES
# ======================

edades = [

    "18-20",
    "21-30",
    "31-40",
    "41-50",
    "51-60"

]

emociones = [

    "anger",
    "contempt",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
    "surprised"

]

# ======================
# GUARDAR CORRECCIONES
# ======================

def guardar_correccion(
    ruta_imagen,
    edad=None,
    emocion=None
):

    if edad:

        carpeta = os.path.join(
            "correcciones",
            "edad",
            edad
        )

        os.makedirs(
            carpeta,
            exist_ok=True
        )

        shutil.copy2(
            ruta_imagen,
            carpeta
        )

    if emocion:

        carpeta = os.path.join(
            "correcciones",
            "emocion",
            emocion
        )

        os.makedirs(
            carpeta,
            exist_ok=True
        )

        shutil.copy2(
            ruta_imagen,
            carpeta
        )

# ======================
# CARGAR IMAGEN
# ======================

ruta = input(
    "Ruta imagen: "
)

img = Image.open(ruta)

img = img.convert("RGB")

img = img.resize(
    (IMG_SIZE,IMG_SIZE)
)

img = np.array(img)

img = img / 255.0

img = np.expand_dims(
    img,
    axis=0
)

# ======================
# PREDICCION EDAD
# ======================

pred_edad = modelo_edad.predict(
    img,
    verbose=0
)

indice_edad = np.argmax(
    pred_edad
)

edad = edades[
    indice_edad
]

confianza_edad = np.max(
    pred_edad
)*100

# ======================
# PREDICCION EMOCION
# ======================

pred_emocion = modelo_emocion.predict(
    img,
    verbose=0
)

indice_emocion = np.argmax(
    pred_emocion
)

emocion = emociones[
    indice_emocion
]

confianza_emocion = np.max(
    pred_emocion
)*100

# ======================
# RESULTADOS
# ======================

print("\n================")

print("RESULTADO IA")

print("================")

print(
f"Edad detectada: {edad}"
)

print(
f"Confianza edad: {confianza_edad:.2f}%"
)

print()

print(
f"Emocion detectada: {emocion}"
)

print(
f"Confianza emocion: {confianza_emocion:.2f}%"
)

print("================")

# ======================
# FEEDBACK
# ======================

respuesta = input(

"\n¿Resultado correcto? (s/n): "

)

if respuesta.lower()=="n":

    print()

    print(
    "¿Que fallo?"
    )

    print(
    "1 - Edad"
    )

    print(
    "2 - Emocion"
    )

    print(
    "3 - Ambas"
    )

    opcion = input(
        "Opcion: "
    )

    edad_correcta=None
    emocion_correcta=None

    if opcion=="1":

        print()

        print(
        "Opciones edad:"
        )

        for e in edades:

            print(
            e
            )

        edad_correcta=input(
        "Edad correcta: "
        )

    elif opcion=="2":

        print()

        print(
        "Opciones emocion:"
        )

        for emo in emociones:

            print(
            emo
            )

        emocion_correcta=input(
        "Emocion correcta: "
        )

    elif opcion=="3":

        print()

        print(
        "Opciones edad:"
        )

        for e in edades:

            print(
            e
            )

        edad_correcta=input(
        "Edad correcta: "
        )

        print()

        print(
        "Opciones emocion:"
        )

        for emo in emociones:

            print(
            emo
            )

        emocion_correcta=input(
        "Emocion correcta: "
        )

    guardar_correccion(
        ruta,
        edad_correcta,
        emocion_correcta
    )

    print()
    print(
    "Correccion guardada"
    )

else:
    print()
    print(
    "Prediccion aceptada"
    )
