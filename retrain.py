import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam

# Hiperparámetros optimizados para forzar cambios drásticos en caliente
IMG_SIZE = 128
BATCH_SIZE = 4 
EPOCHS = 15           # <-- Sido incrementado de 3 a 15 para machacar más veces el error
LEARNING_RATE = 1e-3  # <-- Configuración más agresiva para alterar los pesos con fuerza

def contar_imagenes(carpeta):
    if not os.path.exists(carpeta): 
        return 0
    total = 0
    for subcarpeta in os.listdir(carpeta):
        ruta_sub = os.path.join(carpeta, subcarpeta)
        if os.path.isdir(ruta_sub):
            total += len([f for f in os.listdir(ruta_sub) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))])
    return total

def entrenar_unicamente_fallos(nombre, carpeta_correcciones, carpeta_final_historico, clases_fijas):
    total_fotos = contar_imagenes(carpeta_correcciones)
    
    if total_fotos == 0:
        print(f"ℹ️ [MODELO {nombre.upper()}]: Sin fallos nuevos en '{carpeta_correcciones}'. Saltando...")
        return False

    print(f"\n⚡ [START] Reentrenando {nombre} con {total_fotos} imágenes corregidas.")
    
    # Cargar el modelo base
    modelo = load_model(f"modelo/{nombre}.keras")
    
    # Compilación usando el nuevo ritmo de aprendizaje más veloz
    modelo.compile(optimizer=Adam(learning_rate=LEARNING_RATE), loss='categorical_crossentropy', metrics=['accuracy'])
    
    gen = ImageDataGenerator(rescale=1./255, rotation_range=10, zoom_range=0.1, horizontal_flip=True)
    
    train_data = gen.flow_from_directory(
        carpeta_correcciones,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=min(BATCH_SIZE, total_fotos),
        class_mode="categorical",
        classes=clases_fijas
    )
    
    # Entrenar el modelo
    modelo.fit(train_data, epochs=EPOCHS, verbose=1)
    
    # Guardar sobreescribiendo el modelo del núcleo
    modelo.save(f"modelo/{nombre}.keras")
    print(f"✅ [OK] Modelo '{nombre}' actualizado en el almacenamiento físico.")

    # --- NUEVO AJUSTE: BLOQUE DE BORRADO ELIMINADO ---
    print(f"📌 [CONSERVADO] Imágenes protegidas en '{carpeta_correcciones}' para el reentrenamiento acumulativo.")
    return True
