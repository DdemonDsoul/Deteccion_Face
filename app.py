import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from tkinter import messagebox

from tensorflow.keras.models import load_model
from PIL import Image, ImageTk

import numpy as np
import os
import shutil
import subprocess

IMG = 128

# ======================
# MODELOS
# ======================
try:
    modelo_edad = load_model("modelo/edad.keras")
    modelo_emocion = load_model("modelo/emocion.keras")
    print("✅ Modelos cargados correctamente")
except Exception as e:
    print(f"❌ Error cargando modelos: {e}")
    modelo_edad = None
    modelo_emocion = None

EDADES = [
    "18-20",
    "21-30",
    "31-40",
    "41-50",
    "51-60"
]

EMOCIONES = [
    "anger",
    "contempt",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
    "surprised"
]

imagen_actual = None

# ======================
# VENTANA PRINCIPAL
# ======================
ventana = tk.Tk()
ventana.title("Detector IA - Edad y Emoción")
ventana.geometry("1100x650")
ventana.configure(bg="#18181B")
ventana.resizable(False, False)

# Variables
edad_detectada = ""
emocion_detectada = ""
edad_var = tk.StringVar()
emocion_var = tk.StringVar()
estado = tk.StringVar()
estado.set("✅ Sistema listo")

# ======================
# ESTILOS
# ======================
style = ttk.Style()
style.theme_use("clam")
style.configure("TButton", 
                font=("Segoe UI", 11),
                padding=10)
style.configure("TLabel", 
                background="#18181B",
                foreground="white",
                font=("Segoe UI", 10))
style.configure("TCombobox",
                fieldbackground="#27272A",
                background="#27272A",
                foreground="white")

# ======================
# CONTENEDOR PRINCIPAL (izquierda y derecha)
# ======================
main_container = tk.Frame(ventana, bg="#18181B")
main_container.pack(fill="both", expand=True, padx=20, pady=15)

# ======================
# COLUMNA IZQUIERDA - IMAGEN
# ======================
left_frame = tk.Frame(main_container, bg="#18181B")
left_frame.pack(side="left", fill="both", expand=True, padx=(0, 10))

# Título
titulo = tk.Label(left_frame, 
                  text="DETECTOR IA", 
                  font=("Segoe UI", 20, "bold"),
                  fg="#3B82F6",
                  bg="#18181B")
titulo.pack(pady=(0, 15))

# Subtítulo
subtitulo = tk.Label(left_frame,
                     text="Edad y Emoción",
                     font=("Segoe UI", 14),
                     fg="#9CA3AF",
                     bg="#18181B")
subtitulo.pack(pady=(0, 20))

# Panel de imagen
panel_imagen = tk.Frame(left_frame,
                        bg="#27272A",
                        width=400,
                        height=400,
                        relief="solid",
                        borderwidth=2)
panel_imagen.pack()
panel_imagen.pack_propagate(False)

imagen_label = tk.Label(panel_imagen,
                        bg="#27272A",
                        text="📷\n\nSin imagen",
                        font=("Segoe UI", 14),
                        fg="#6B7280")
imagen_label.pack(expand=True, fill="both")

# ======================
# COLUMNA DERECHA - FUNCIONALIDAD
# ======================
right_frame = tk.Frame(main_container, bg="#18181B")
right_frame.pack(side="right", fill="both", expand=True, padx=(10, 0))

# ===== RESULTADOS =====
resultados_frame = tk.LabelFrame(right_frame,
                                 text=" RESULTADOS DEL ANÁLISIS ",
                                 font=("Segoe UI", 12, "bold"),
                                 bg="#18181B",
                                 fg="#3B82F6",
                                 padx=15,
                                 pady=15)
resultados_frame.pack(fill="x", pady=(0, 20))

# Edad
tk.Label(resultados_frame,
         text="📊 EDAD DETECTADA",
         font=("Segoe UI", 12, "bold"),
         fg="white",
         bg="#18181B").pack(anchor="w", pady=(0, 5))

edad_frame = tk.Frame(resultados_frame, bg="#27272A", relief="solid", borderwidth=1)
edad_frame.pack(fill="x", pady=(0, 15))
tk.Label(edad_frame,
         textvariable=edad_var,
         font=("Segoe UI", 16, "bold"),
         fg="#60A5FA",
         bg="#27272A",
         padx=15,
         pady=10).pack(anchor="w")

# Emoción
tk.Label(resultados_frame,
         text="😊 EMOCIÓN DETECTADA",
         font=("Segoe UI", 12, "bold"),
         fg="white",
         bg="#18181B").pack(anchor="w", pady=(0, 5))

emocion_frame = tk.Frame(resultados_frame, bg="#27272A", relief="solid", borderwidth=1)
emocion_frame.pack(fill="x")
tk.Label(emocion_frame,
         textvariable=emocion_var,
         font=("Segoe UI", 16, "bold"),
         fg="#60A5FA",
         bg="#27272A",
         padx=15,
         pady=10).pack(anchor="w")

# ===== BOTONES PRINCIPALES =====
botones_frame = tk.Frame(right_frame, bg="#18181B")
botones_frame.pack(fill="x", pady=(0, 20))

def crear_boton(texto, comando, color, icono=""):
    btn = tk.Button(botones_frame,
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

crear_boton("Seleccionar Imagen", lambda: analizar(), "#2563EB", "📁")
crear_boton("Reportar Fallo", fallo_ia, "#DC2626", "⚠️")
crear_boton("Reentrenar IA", retrain, "#8B5CF6", "🔄")

# ===== CORRECCIONES RÁPIDAS =====
correcciones_frame = tk.LabelFrame(right_frame,
                                    text=" CORRECCIONES RÁPIDAS ",
                                    font=("Segoe UI", 11, "bold"),
                                    bg="#18181B",
                                    fg="#9CA3AF",
                                    padx=10,
                                    pady=10)
correcciones_frame.pack(fill="x")

# Información
info_label = tk.Label(correcciones_frame,
                      text="ℹ️ Selecciona una imagen primero\npara habilitar correcciones",
                      font=("Segoe UI", 9),
                      bg="#18181B",
                      fg="#6B7280",
                      justify="left")
info_label.pack(anchor="w", pady=5)

# ======================
# FUNCIONES
# ======================
def analizar():
    global imagen_actual, edad_detectada, emocion_detectada
    
    if modelo_edad is None or modelo_emocion is None:
        messagebox.showerror("Error", "Los modelos no están cargados correctamente")
        return

    ruta = filedialog.askopenfilename(
        filetypes=[("Imagen", "*.jpg *.jpeg *.png *.webp")]
    )

    if not ruta:
        return

    imagen_actual = ruta

    # Mostrar imagen
    img = Image.open(ruta)
    mostrar = img.copy()
    mostrar.thumbnail((350, 350))
    foto = ImageTk.PhotoImage(mostrar)
    imagen_label.config(image=foto, text="")
    imagen_label.image = foto

    # Procesar para IA
    ia = img.convert("RGB")
    ia = ia.resize((IMG, IMG))
    ia = np.array(ia)
    ia = ia / 255.0
    ia = np.expand_dims(ia, axis=0)

    # Predicciones
    try:
        edad = modelo_edad.predict(ia, verbose=0)
        emo = modelo_emocion.predict(ia, verbose=0)

        edad_detectada = EDADES[np.argmax(edad)]
        emocion_detectada = EMOCIONES[np.argmax(emo)]

        edad_var.set(f"{edad_detectada}\nConfianza: {np.max(edad)*100:.1f}%")
        emocion_var.set(f"{emocion_detectada}\nConfianza: {np.max(emo)*100:.1f}%")
        estado.set("✅ Análisis completado")
        
        # Actualizar info de correcciones
        info_label.config(text="✅ Imagen analizada\nPuedes reportar fallos si es necesario", fg="#10B981")
    except Exception as e:
        estado.set(f"❌ Error: {str(e)}")
        messagebox.showerror("Error", f"Error en la predicción: {str(e)}")

def guardar():
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

    # Centrar el popup
    popup.transient(ventana)
    popup.grab_set()

    tk.Label(popup,
             text="Corrección de Predicción",
             font=("Segoe UI", 14, "bold"),
             fg="#3B82F6",
             bg="#27272A").pack(pady=15)

    fallo = tk.StringVar()
    fallo.set("Edad")

    tk.Label(popup,
             text="¿Qué falló?",
             font=("Segoe UI", 11),
             bg="#27272A",
             fg="white").pack(pady=(10, 5))

    fallo_combo = ttk.Combobox(popup,
                               textvariable=fallo,
                               values=["Edad", "Emoción", "Ambas"],
                               state="readonly",
                               width=20)
    fallo_combo.pack(pady=5)

    # Edad correcta
    tk.Label(popup,
             text="Edad correcta:",
             font=("Segoe UI", 11),
             bg="#27272A",
             fg="white").pack(pady=(10, 5))

    edad_box = ttk.Combobox(popup,
                            values=EDADES,
                            state="readonly",
                            width=20)
    edad_box.pack(pady=5)
    edad_box.current(0)

    # Emoción correcta
    tk.Label(popup,
             text="Emoción correcta:",
             font=("Segoe UI", 11),
             bg="#27272A",
             fg="white").pack(pady=(10, 5))

    emo_box = ttk.Combobox(popup,
                           values=EMOCIONES,
                           state="readonly",
                           width=20)
    emo_box.pack(pady=5)
    emo_box.current(0)

    # Botones
    btn_frame = tk.Frame(popup, bg="#27272A")
    btn_frame.pack(pady=20)

    tk.Button(btn_frame,
              text="Guardar",
              command=guardar,
              font=("Segoe UI", 10, "bold"),
              bg="#10B981",
              fg="white",
              padx=20,
              pady=5,
              cursor="hand2").pack(side="left", padx=5)

    tk.Button(btn_frame,
              text="Cancelar",
              command=popup.destroy,
              font=("Segoe UI", 10),
              bg="#6B7280",
              fg="white",
              padx=20,
              pady=5,
              cursor="hand2").pack(side="left", padx=5)

def retrain():
    if imagen_actual is None:
        messagebox.showwarning("Aviso", "Se recomienda tener al menos una corrección antes de reentrenar")
    
    estado.set("🔄 Reentrenando IA...")
    ventana.update()
    
    try:
        subprocess.run(["python", "retrain.py"], check=True)
        estado.set("✅ IA actualizada correctamente")
        messagebox.showinfo("Correcto", "Modelos reentrenados y actualizados")
    except Exception as e:
        estado.set("❌ Error en reentrenamiento")
        messagebox.showerror("Error", f"No se pudo ejecutar retrain.py\n{str(e)}")

# ======================
# BARRA DE ESTADO
# ======================
status_bar = tk.Frame(ventana, bg="#111827", height=35)
status_bar.pack(side="bottom", fill="x")

status_label = tk.Label(status_bar,
                        textvariable=estado,
                        bg="#111827",
                        fg="#9CA3AF",
                        font=("Segoe UI", 9),
                        padx=10,
                        pady=8,
                        anchor="w")
status_label.pack(fill="x")

# Inicializar mensaje de bienvenida
edad_var.set("Esperando imagen...")
emocion_var.set("Esperando imagen...")

ventana.mainloop()
