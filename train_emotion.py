from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout

IMG_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 10

# Generador imágenes
generador = ImageDataGenerator(

    rescale=1./255,

    validation_split=0.2,

    rotation_range=10,

    zoom_range=0.1,

    horizontal_flip=True

)

# Entrenamiento
train = generador.flow_from_directory(

    "datasets/emociones_procesado",

    target_size=(IMG_SIZE,IMG_SIZE),

    batch_size=BATCH_SIZE,

    class_mode="categorical",

    subset="training"

)

# Validación
validacion = generador.flow_from_directory(

    "datasets/emociones_procesado",

    target_size=(IMG_SIZE,IMG_SIZE),

    batch_size=BATCH_SIZE,

    class_mode="categorical",

    subset="validation"

)

print("\nEmociones detectadas:")
print(train.class_indices)

NUM_CLASES = len(train.class_indices)

modelo = Sequential([

    Input(
        shape=(IMG_SIZE,IMG_SIZE,3)
    ),

    Conv2D(
        32,
        (3,3),
        activation="relu"
    ),

    MaxPooling2D(2,2),

    Conv2D(
        64,
        (3,3),
        activation="relu"
    ),

    MaxPooling2D(2,2),

    Conv2D(
        128,
        (3,3),
        activation="relu"
    ),

    MaxPooling2D(2,2),

    Flatten(),

    Dense(
        128,
        activation="relu"
    ),

    Dropout(0.5),

    Dense(
        NUM_CLASES,
        activation="softmax"
    )

])

modelo.compile(

    optimizer="adam",

    loss="categorical_crossentropy",

    metrics=["accuracy"]

)

modelo.summary()

print("\nEntrenando emociones...\n")

historial = modelo.fit(

    train,

    validation_data=validacion,

    epochs=EPOCHS

)

modelo.save(

    "modelo/emocion.keras"

)

print(
"\nModelo emociones guardado"
)
