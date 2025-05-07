import customtkinter as ctk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
from deepface import DeepFace
import threading
import numpy as np

# Configuración global
ctk.set_appearance_mode("light")  # Cambiar a modo claro para tener el fondo blanco
ctk.set_default_color_theme("green")

class Inicio(ctk.CTkFrame):
    def __init__(self, master):
        super().__init__(master)
        self.master = master

        # Fondo blanco para el frame
        self.configure(bg_color='white')

        # Cargar imagen del logo
        logo_img = Image.open("emociones2.png")  # Asegúrate de poner la ruta correcta
        logo_img = logo_img.resize((300, 300))  # Ajusta el tamaño del logo si es necesario
        logo_img_tk = ImageTk.PhotoImage(logo_img)

        # Crear una etiqueta para mostrar el logo
        logo_label = ctk.CTkLabel(self, image=logo_img_tk)
        logo_label.image = logo_img_tk  # Guardar una referencia de la imagen
        logo_label.pack(pady=20)

        # Título de la ventana de inicio (corregido: usando `text_color` en lugar de `fg`)
        ctk.CTkLabel(self, text="EmotionSense AI", font=("Arial Black", 28), text_color='black').pack(pady=40)

        # Botones con colores personalizados
        ctk.CTkButton(self, text="Iniciar Análisis", command=self.ir_a_analisis, width=200, 
                      fg_color="#6a5acd", hover_color="#483d8b", border_color="black").pack(pady=20)
        ctk.CTkButton(self, text="Salir", command=self.master.quit, width=200, 
                      fg_color="#6a5acd", hover_color="#483d8b", border_color="black").pack(pady=10)

    def ir_a_analisis(self):
        self.master.mostrar_ventana(Analisis)

class Analisis(ctk.CTkFrame):
    def __init__(self, master):
        super().__init__(master)
        self.master = master
        self.camera_active = False
        self.cap = None
        self.analizando = False

        self.emotion_translations = {
            'angry': 'Enojo',
            'happy': 'Feliz',
            'sad': 'Triste',
        }

        ctk.CTkLabel(self, text="Análisis Facial", font=("Arial", 22), text_color='black').pack(pady=10)

        btn_frame = ctk.CTkFrame(self)
        btn_frame.pack(pady=5)
        ctk.CTkButton(btn_frame, text="📷 Cámara", command=self.toggle_camera, width=150,
                      fg_color="#6a5acd", hover_color="#483d8b", border_color="black").grid(row=0, column=0, padx=10)
        ctk.CTkButton(btn_frame, text="📂 Cargar Imagen", command=self.load_image, width=150,
                      fg_color="#6a5acd", hover_color="#483d8b", border_color="black").grid(row=0, column=1, padx=10)

        self.image_canvas = ctk.CTkLabel(self, text="", width=640, height=400, corner_radius=12)
        self.image_canvas.pack(pady=10)

        self.result_label = ctk.CTkLabel(self, text="Emoción detectada: ---", font=("Arial", 18))
        self.result_label.pack(pady=10)

        btm_frame = ctk.CTkFrame(self)
        btm_frame.pack(pady=5)
        ctk.CTkButton(btm_frame, text="⬅ Volver", command=self.volver, width=150,
                      fg_color="#6a5acd", hover_color="#483d8b", border_color="black").grid(row=0, column=0, padx=10)
        ctk.CTkButton(btm_frame, text="💾 Guardar Resultado", command=self.save_results, width=180,
                      fg_color="#6a5acd", hover_color="#483d8b", border_color="black").grid(row=0, column=1, padx=10)

    def volver(self):
        if self.camera_active:
            self.camera_active = False
            self.cap.release()
        self.master.mostrar_ventana(Inicio)

    def toggle_camera(self):
        if self.camera_active:
            self.camera_active = False
            self.cap.release()
            self.image_canvas.configure(image=None, text="Cámara detenida")
        else:
            self.cap = cv2.VideoCapture(0)
            self.camera_active = True
            threading.Thread(target=self.show_camera, daemon=True).start()

    def show_camera(self):
        frame_counter = 0
        while self.camera_active:
            ret, frame = self.cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb).resize((640, 400))
            imgtk = ImageTk.PhotoImage(img)
            self.image_canvas.configure(image=imgtk)
            self.image_canvas.image = imgtk

            frame_counter += 1
            if frame_counter % 30 == 0 and not self.analizando:
                self.analizando = True
                self.result_label.configure(text="Analizando emoción...")
                frame_copy = frame.copy()
                threading.Thread(target=self.analyze_emotion_thread, args=(frame_copy,), daemon=True).start()

    def analyze_emotion_thread(self, frame):
        try:
            result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
            dominant = result[0]['dominant_emotion']
            if dominant.lower() in self.emotion_translations:
                translated = self.emotion_translations[dominant.lower()]
                self.result_label.configure(text=f"Emoción detectada: {translated}")
            else:
                self.result_label.configure(text="Emoción no válida")
        except Exception:
            self.result_label.configure(text="Error detectando emoción")
        self.analizando = False

    def load_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Imágenes", "*.jpg *.jpeg *.png")])
        if file_path:
            img = Image.open(file_path).resize((640, 400))
            img_rgb = img.convert("RGB")
            img_array = np.array(img_rgb)

            imgtk = ImageTk.PhotoImage(img)
            self.image_canvas.configure(image=imgtk)
            self.image_canvas.image = imgtk

            self.result_label.configure(text="Analizando emoción...")
            threading.Thread(target=self.analyze_emotion_thread, args=(img_array,), daemon=True).start()

    def save_results(self):
        with open("reporte_emociones.txt", "a", encoding="utf-8") as f:
            f.write(self.result_label.cget("text") + "\n")
        self.result_label.configure(text="Resultado guardado.")

class EmotionSenseApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("EmotionSense AI")

        # Centrar ventana con tamaño fijo 950x720
        ancho = 950
        alto = 720
        x = (self.winfo_screenwidth() // 2) - (ancho // 2)
        y = (self.winfo_screenheight() // 2) - (alto // 2)
        self.geometry(f"{ancho}x{alto}+{x}+{y}")

        self.resizable(False, False)

        self.frames = {}
        for F in (Inicio, Analisis):
            frame = F(self)
            self.frames[F] = frame
            frame.place(relwidth=1, relheight=1)

        self.mostrar_ventana(Inicio)

    def mostrar_ventana(self, contenedor):
        frame = self.frames[contenedor]
        frame.tkraise()

if __name__ == "__main__":
    app = EmotionSenseApp()
    app.mainloop()
