# train_age.py
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'   # Bloquea alertas densas en la consola
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Evita crasheos por optimizaciones de CPU en Windows

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from config import IMG_SIZE, BATCH_SIZE

EPOCHS = 15

generador = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=15,
    zoom_range=0.15,
    horizontal_flip=True
)

train = generador.flow_from_directory(
    "datasets/edades",
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training"
)

validacion = generador.flow_from_directory(
    "datasets/edades",
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation"
)

print("\nCategorías detectadas por Keras (orden alfabético estricto):")
print(train.class_indices)
NUM_CLASES = len(train.class_indices)

modelo = Sequential([
    Input(shape=(IMG_SIZE, IMG_SIZE, 3)),
    
    Conv2D(32, (3,3), activation="relu"),
    BatchNormalization(),
    MaxPooling2D(2,2),
    Dropout(0.25),
    
    Conv2D(64, (3,3), activation="relu"),
    BatchNormalization(),
    MaxPooling2D(2,2),
    Dropout(0.25),
    
    Conv2D(128, (3,3), activation="relu"),
    BatchNormalization(),
    MaxPooling2D(2,2),
    Dropout(0.3),
    
    Flatten(),
    Dense(256, activation="relu"),
    BatchNormalization(),
    Dropout(0.5),
    Dense(NUM_CLASES, activation="softmax")
])

# Ajuste: Optimizador suavizado para estabilizar picos en la gráfica
optimizador = Adam(learning_rate=0.0001)
modelo.compile(optimizer=optimizador, loss="categorical_crossentropy", metrics=["accuracy"])

# Ajuste: Freno automático si la pérdida de validación deja de mejorar
monitoreo = EarlyStopping(
    monitor='val_loss', 
    patience=4, 
    restore_best_weights=True
)

print("\nEntrenando modelo de EDADES con optimización suavizada...\n")
historial = modelo.fit(
    train, 
    validation_data=validacion, 
    epochs=EPOCHS, 
    callbacks=[monitoreo]
)

modelo.save("modelo/edad.keras")
print("\nModelo de edades guardado correctamente")

# Graficar resultados
acc = historial.history['accuracy']
val_acc = historial.history['val_accuracy']
loss = historial.history['loss']
val_loss = historial.history['val_loss']
epochs_range = range(len(acc))

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Entrenamiento')
plt.plot(epochs_range, val_acc, label='Validación')
plt.legend(loc='lower right')
plt.title('Precisión (Accuracy)')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Entrenamiento')
plt.plot(epochs_range, val_loss, label='Validación')
plt.legend(loc='upper right')
plt.title('Pérdida (Loss)')
plt.show()
