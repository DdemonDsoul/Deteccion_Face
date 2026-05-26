import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from tensorflow.keras.models import load_model
from PIL import Image, ImageTk
import numpy as np
import os
import shutil
import threading  # Lógica multihilo nativa

# Importaciones del ecosistema unificado
from config import MAPEO_EDADES, MAPEO_EMOCIONES, EDADES_CARPETAS, EMOCIONES_CARPETAS
from preproces import procesar_imagen
from retrain import entrenar_unicamente_fallos, contar_imagenes  # <--- IMPORTACIÓN DIRECTA NATIVA

try:
    modelo_edad = load_model("modelo/edad.keras")
    modelo_emocion = load_model("modelo/emocion.keras")
    print("✅ Modelos de Edad y Emoción cargados correctamente")
except Exception as e:
    print(f"❌ Error cargando modelos: {e}")
    modelo_edad = None
    modelo_emocion = None

imagen_actual = None
popup = None
fallo = None
edad_box = None
emo_box = None

def analizar():
    global imagen_actual, edad_detectada, emocion_detectada
    
    if modelo_edad is None or modelo_emocion is None:
        messagebox.showerror("Error", "Los modelos no están cargados correctamente")
        return

    ruta = filedialog.askopenfilename(filetypes=[("Imagen", "*.jpg *.jpeg *.png *.webp")])
    if not ruta:
        return

    imagen_actual = ruta

    img = Image.open(ruta)
    mostrar = img.copy()
    mostrar.thumbnail((350, 350))
    foto = ImageTk.PhotoImage(mostrar)
    imagen_label.config(image=foto, text="")
    imagen_label.image = foto

    try:
        ia = procesar_imagen(ruta)

        edad_pred = modelo_edad.predict(ia, verbose=0)
        emo_pred = modelo_emocion.predict(ia, verbose=0)

        id_edad_carpeta = EDADES_CARPETAS[np.argmax(edad_pred)]
        id_emo_carpeta = EMOCIONES_CARPETAS[np.argmax(emo_pred)]

        edad_detectada = MAPEO_EDADES[id_edad_carpeta]
        emocion_detectada = MAPEO_EMOCIONES[id_emo_carpeta]

        edad_var.set(f"{edad_detectada}\nConfianza: {np.max(edad_pred)*100:.1f}%")
        emocion_var.set(f"{emocion_detectada}\nConfianza: {np.max(emo_pred)*100:.1f}%")
        estado.set("✅ Análisis completado")
        info_label.config(text="✅ Imagen analizada\nPuedes reportar fallos si es necesario", fg="#10B981")
    except Exception as e:
        estado.set(f"❌ Error: {str(e)}")
        messagebox.showerror("Error", f"Error en la predicción: {str(e)}")

def guardar():
    global popup, fallo, edad_box, emo_box

    fallo_valor = fallo.get()
    edad_valor = edad_box.get()
    emo_valor = emo_box.get()

    if fallo_valor in ["Edad", "Ambas"]:
        carpeta = os.path.join("correcciones", "edad", edad_valor)
        os.makedirs(carpeta, exist_ok=True)
        shutil.copy2(imagen_actual, carpeta)

    if fallo_valor in ["Emoción", "Ambas"]:
        carpeta = os.path.join("correcciones", "emocion", emo_valor)
        os.makedirs(carpeta, exist_ok=True)
        shutil.copy2(imagen_actual, carpeta)

    popup.destroy()
    messagebox.showinfo("Listo", "Corrección guardada correctamente")
    estado.set("✅ Corrección guardada")

def fallo_ia():
    global popup, fallo, edad_box, emo_box

    if imagen_actual is None:
        messagebox.showwarning("Aviso", "Primero debes analizar una imagen")
        return

    popup = tk.Toplevel()
    popup.title("Corregir IA")
    popup.geometry("400x350")
    popup.configure(bg="#27272A")
    popup.resizable(False, False)
    popup.transient(ventana)
    popup.grab_set()

    tk.Label(popup, text="Corrección de Predicción", font=("Segoe UI", 14, "bold"), fg="#3B82F6", bg="#27272A").pack(pady=15)

    fallo = tk.StringVar()
    fallo.set("Edad")

    tk.Label(popup, text="¿Qué falló?", font=("Segoe UI", 11), bg="#27272A", fg="white").pack(pady=(10, 5))
    fallo_combo = ttk.Combobox(popup, textvariable=fallo, values=["Edad", "Emoción", "Ambas"], state="readonly", width=20)
    fallo_combo.pack(pady=5)

    tk.Label(popup, text="Edad correcta:", font=("Segoe UI", 11), bg="#27272A", fg="white").pack(pady=(10, 5))
    edad_box = ttk.Combobox(popup, values=EDADES_CARPETAS, state="readonly", width=20)
    edad_box.pack(pady=5)
    edad_box.current(0)

    tk.Label(popup, text="Emoción correcta:", font=("Segoe UI", 11), bg="#27272A", fg="white").pack(pady=(10, 5))
    emo_box = ttk.Combobox(popup, values=EMOCIONES_CARPETAS, state="readonly", width=20)
    emo_box.pack(pady=5)
    emo_box.current(0)

    btn_frame = tk.Frame(popup, bg="#27272A")
    btn_frame.pack(pady=20)

    tk.Button(btn_frame, text="Guardar", command=guardar, font=("Segoe UI", 10, "bold"), bg="#10B981", fg="white", padx=20, pady=5, cursor="hand2").pack(side="left", padx=5)
    tk.Button(btn_frame, text="Cancelar", command=popup.destroy, font=("Segoe UI", 10), bg="#6B7280", fg="white", padx=20, pady=5, cursor="hand2").pack(side="left", padx=5)

def retrain():
    total_cambios = contar_imagenes("correcciones/edad") + contar_imagenes("correcciones/emocion")
    if total_cambios == 0:
        messagebox.showinfo("Información", "No hay correcciones nuevas guardadas para entrenar.")
        return

    def worker_entrenamiento():
        global modelo_edad, modelo_emocion
        try:
            hubo_cambio_edad = False
            hubo_cambio_emo = False

            if contar_imagenes("correcciones/edad") > 0:
                estado.set("🔄 Reentrenando Edades...")
                hubo_cambio_edad = entrenar_unicamente_fallos("edad", "correcciones/edad", "datasets/edades", EDADES_CARPETAS)

            if contar_imagenes("correcciones/emocion") > 0:
                estado.set("🔄 Reentrenando Emociones...")
                hubo_cambio_emo = entrenar_unicamente_fallos("emocion", "correcciones/emocion", "datasets/emociones_procesado", EMOCIONES_CARPETAS)
            
            # --- LIMPIEZA Y RECARGA EN CALIENTE CONTRA CACHÉ ---
            estado.set("🔄 Liberando caché de memoria RAM...")
            
            # 1. Forzar a Keras a olvidar los grafos internos antiguos del hilo de ejecución
            from tensorflow.keras.backend import clear_session
            clear_session()
            
            # 2. Forzar la recarga física desde el disco a las variables del hilo principal
            if hubo_cambio_edad:
                print("🔄 Cargando pesos actualizados de Edad...")
                modelo_edad = load_model("modelo/edad.keras")
            if hubo_cambio_emo:
                print("🔄 Cargando pesos actualizados de Emoción...")
                modelo_emocion = load_model("modelo/emocion.keras")
            
            estado.set("✅ IA totalmente actualizada")
            messagebox.showinfo("Correcto", "¡Modelos reentrenados y sincronizados perfectamente en memoria de forma segura!")
            
            # Solicitamos volver a procesar la imagen para limpiar estados visuales intermedios
            edad_var.set("Pesos actualizados. Analiza de nuevo.")
            emocion_var.set("Pesos actualizados. Analiza de nuevo.")
            
        except Exception as e:
            estado.set("❌ Error en reentrenamiento")
            messagebox.showerror("Error", f"Error interno durante la ejecución del reentrenamiento:\n{str(e)}")

    threading.Thread(target=worker_entrenamiento, daemon=True).start()

def crear_boton(contenedor, texto, comando, color, icono=""):
    btn = tk.Button(contenedor,
                    text=f"{icono} {texto}" if icono else texto,
                    command=comando,
                    font=("Segoe UI", 11, "bold"),
                    bg=color,
                    fg="white",
                    cursor="hand2",
                    relief="raised",
                    bd=2,
                    padx=15,
                    pady=10)
    btn.pack(fill="x", pady=5)
    return btn

# --- INTERFAZ GRÁFICA PRINCIPAL ---
ventana = tk.Tk()
ventana.title("Detector IA - Edad y Emoción")
ventana.geometry("1100x650")
ventana.configure(bg="#18181B")
ventana.resizable(False, False)

edad_detectada = ""
emocion_detectada = ""
edad_var = tk.StringVar()
emocion_var = tk.StringVar()
estado = tk.StringVar()
estado.set("✅ Sistema listo")

style = ttk.Style()
style.theme_use("clam")
style.configure("TButton", font=("Segoe UI", 11), padding=10)
style.configure("TLabel", background="#18181B", foreground="white", font=("Segoe UI", 10))
style.configure("TCombobox", fieldbackground="#27272A", background="#27272A", foreground="white")

main_container = tk.Frame(ventana, bg="#18181B")
main_container.pack(fill="both", expand=True, padx=20, pady=15)

left_frame = tk.Frame(main_container, bg="#18181B")
left_frame.pack(side="left", fill="both", expand=True, padx=(0, 10))

tk.Label(left_frame, text="DETECTOR IA", font=("Segoe UI", 20, "bold"), fg="#3B82F6", bg="#18181B").pack(pady=(0, 15))
tk.Label(left_frame, text="Edad y Emoción", font=("Segoe UI", 14), fg="#9CA3AF", bg="#18181B").pack(pady=(0, 20))

panel_imagen = tk.Frame(left_frame, bg="#27272A", width=400, height=400, relief="solid", borderwidth=2)
panel_imagen.pack()
panel_imagen.pack_propagate(False)

imagen_label = tk.Label(panel_imagen, bg="#27272A", text="📷\n\nSin imagen", font=("Segoe UI", 14), fg="#6B7280")
imagen_label.pack(expand=True, fill="both")

right_frame = tk.Frame(main_container, bg="#18181B")
right_frame.pack(side="right", fill="both", expand=True, padx=(10, 0))

resultados_frame = tk.LabelFrame(right_frame, text=" RESULTADOS DEL ANÁLISIS ", font=("Segoe UI", 12, "bold"), bg="#18181B", fg="#3B82F6", padx=15, pady=15)
resultados_frame.pack(fill="x", pady=(0, 20))

tk.Label(resultados_frame, text="📊 EDAD DETECTADA", font=("Segoe UI", 12, "bold"), fg="white", bg="#18181B").pack(anchor="w", pady=(0, 5))
edad_frame = tk.Frame(resultados_frame, bg="#27272A", relief="solid", borderwidth=1)
edad_frame.pack(fill="x", pady=(0, 15))
tk.Label(edad_frame, textvariable=edad_var, font=("Segoe UI", 16, "bold"), fg="#60A5FA", bg="#27272A", padx=15, pady=10).pack(anchor="w")

tk.Label(resultados_frame, text="😊 EMOCIÓN DETECTADA", font=("Segoe UI", 12, "bold"), fg="white", bg="#18181B").pack(anchor="w", pady=(0, 5))
emocion_frame = tk.Frame(resultados_frame, bg="#27272A", relief="solid", borderwidth=1)
emocion_frame.pack(fill="x")
tk.Label(emocion_frame, textvariable=emocion_var, font=("Segoe UI", 16, "bold"), fg="#60A5FA", bg="#27272A", padx=15, pady=10).pack(anchor="w")

botones_frame = tk.Frame(right_frame, bg="#18181B")
botones_frame.pack(fill="x", pady=(0, 20))

crear_boton(botones_frame, "Seleccionar Imagen", analizar, "#2563EB", "📁")
crear_boton(botones_frame, "Reportar Fallo", fallo_ia, "#DC2626", "⚠️")
crear_boton(botones_frame, "Reentrenar IA", retrain, "#8B5CF6", "🔄")

correcciones_frame = tk.LabelFrame(right_frame, text=" CORRECCIONES RÁPIDAS ", font=("Segoe UI", 11, "bold"), bg="#18181B", fg="#9CA3AF", padx=10, pady=10)
correcciones_frame.pack(fill="x")

info_label = tk.Label(correcciones_frame, text="ℹ️ Selecciona una imagen primero\npara habilitar correcciones", font=("Segoe UI", 9), bg="#18181B", fg="#6B7280", justify="left")
info_label.pack(anchor="w", pady=5)

status_bar = tk.Frame(ventana, bg="#111827", height=35)
status_bar.pack(side="bottom", fill="x")
status_label = tk.Label(status_bar, textvariable=estado, bg="#111827", fg="#9CA3AF", font=("Segoe UI", 9), padx=10, pady=8, anchor="w")
status_label.pack(fill="x")

edad_var.set("Esperando imagen...")
emocion_var.set("Esperando imagen...")

ventana.mainloop()
