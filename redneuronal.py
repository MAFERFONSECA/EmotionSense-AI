import customtkinter as ctk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
from deepface import DeepFace
import threading
import numpy as np

# Configuración inicial de la app
ctk.set_appearance_mode("dark")  # Opciones: "System", "Light", "Dark"
ctk.set_default_color_theme("green")  # Opciones: "blue", "green", "dark-blue"

class EmotionApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("EmotionSense AI")
        self.geometry("950x720")
        self.resizable(False, False)

        self.camera_active = False
        self.cap = None

        # Diccionario para traducir emociones
        self.emotion_translations = {
            'angry': 'Enojo',
            'disgust': 'Asco',
            'fear': 'Miedo',
            'happy': 'Feliz',
            'sad': 'Triste',
            'surprise': 'Sorpresa',
            'neutral': 'Neutral'
        }

        # --- Título principal ---
        self.main_label = ctk.CTkLabel(self, text="EmotionSense AI", font=("Arial Black", 28))
        self.main_label.pack(pady=20)

        # --- Botones principales ---
        self.button_frame = ctk.CTkFrame(self)
        self.button_frame.pack(pady=10)

        self.btn_camera = ctk.CTkButton(self.button_frame, text="📷 Iniciar Cámara", width=180, command=self.toggle_camera)
        self.btn_camera.grid(row=0, column=0, padx=10, pady=10)

        self.btn_load = ctk.CTkButton(self.button_frame, text="📂 Cargar Imagen", width=180, command=self.load_image)
        self.btn_load.grid(row=0, column=1, padx=10, pady=10)

        # --- Área de imagen ---
        self.image_canvas = ctk.CTkLabel(self, text="", width=640, height=400, corner_radius=12)
        self.image_canvas.pack(pady=20)

        # --- Resultado ---
        self.result_label = ctk.CTkLabel(self, text="Emoción detectada: ---", font=("Arial", 20))
        self.result_label.pack(pady=10)

        # --- Botones secundarios ---
        self.secondary_frame = ctk.CTkFrame(self)
        self.secondary_frame.pack(pady=10)

        self.btn_save = ctk.CTkButton(self.secondary_frame, text="💾 Guardar Resultado", width=180, command=self.save_results)
        self.btn_save.grid(row=0, column=0, padx=10)

        self.btn_exit = ctk.CTkButton(self.secondary_frame, text="❌ Salir", width=180, command=self.quit)
        self.btn_exit.grid(row=0, column=1, padx=10)

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
        while self.camera_active:
            ret, frame = self.cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb).resize((640, 400))
            imgtk = ImageTk.PhotoImage(img)

            self.image_canvas.configure(image=imgtk)
            self.image_canvas.image = imgtk

            # Analizar emoción cada 20 frames aprox
            if self.cap.get(cv2.CAP_PROP_POS_FRAMES) % 20 == 0:
                try:
                    result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
                    dominant = result[0]['dominant_emotion']
                    translated = self.emotion_translations.get(dominant.lower(), dominant)
                    self.result_label.configure(text=f"Emoción detectada: {translated}")
                except Exception:
                    self.result_label.configure(text="Error detectando emoción")

    def load_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Imágenes", "*.jpg *.jpeg *.png")])
        if file_path:
            # Cargar y mostrar imagen
            img = Image.open(file_path).resize((640, 400))
            img_rgb = img.convert("RGB")
            img_array = np.array(img_rgb)

            imgtk = ImageTk.PhotoImage(img)
            self.image_canvas.configure(image=imgtk)
            self.image_canvas.image = imgtk

            # Análisis de emoción
            try:
                result = DeepFace.analyze(img_array, actions=['emotion'], enforce_detection=False)
                dominant = result[0]['dominant_emotion']
                translated = self.emotion_translations.get(dominant.lower(), dominant)
                self.result_label.configure(text=f"Emoción detectada: {translated}")
            except Exception:
                self.result_label.configure(text="Error detectando emoción")

    def save_results(self):
        with open("reporte_emociones.txt", "a", encoding="utf-8") as f:
            f.write(self.result_label.cget("text") + "\n")
        self.result_label.configure(text="Resultado guardado.")

# Ejecutar la app
if __name__ == "__main__":
    app = EmotionApp()
    app.mainloop()

