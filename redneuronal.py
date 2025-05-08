import customtkinter as ctk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
from deepface import DeepFace
import threading
import numpy as np
from datetime import datetime

ctk.set_appearance_mode("light")
ctk.set_default_color_theme("green")

class Inicio(ctk.CTkFrame):
    def __init__(self, master):
        super().__init__(master)
        self.master = master
        self.configure(bg_color='white')

        logo_img = Image.open("logo3.png").resize((300, 300))
        logo_img_tk = ImageTk.PhotoImage(logo_img)

        logo_label = ctk.CTkLabel(self, image=logo_img_tk, text="")
        logo_label.image = logo_img_tk
        logo_label.pack(pady=20)

        ctk.CTkLabel(self, text="EmotionSense AI :)", font=("Segoe UI Black", 50), text_color='#6A5ACD').pack(pady=40)

        ctk.CTkButton(self, text="Iniciar Análisis", command=self.ir_a_analisis, width=200,
                      fg_color="#3ba200", hover_color="#297200", border_color="black").pack(pady=10)

        ctk.CTkButton(self, text="📜 Ver Historial", command=self.ir_a_historial, width=200,
                      fg_color="#6a5acd", hover_color="#483d8b", border_color="black").pack(pady=10)

        ctk.CTkButton(self, text="Salir", command=self.master.quit, width=100,
                      fg_color="#a20000", hover_color="#720000", border_color="black").pack(pady=10)

    def ir_a_analisis(self):
        self.master.mostrar_ventana(Analisis)

    def ir_a_historial(self):
        self.master.mostrar_ventana(Historial)

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
                      fg_color="#6a5acd", hover_color="#483d8b").grid(row=0, column=0, padx=10)

        ctk.CTkButton(btn_frame, text="📂 Cargar Imagen", command=self.load_image, width=150,
                      fg_color="#6a5acd", hover_color="#483d8b").grid(row=0, column=1, padx=10)

        ctk.CTkButton(btn_frame, text="📸 Capturar Foto", command=self.capturar_foto, width=150,
                      fg_color="#6a5acd", hover_color="#483d8b").grid(row=0, column=2, padx=10)

        self.image_canvas = ctk.CTkLabel(self, text="", width=640, height=400, corner_radius=12)
        self.image_canvas.pack(pady=10)

        self.result_label = ctk.CTkLabel(self, text="Emoción detectada: ---", font=("Arial", 18))
        self.result_label.pack(pady=10)

        btm_frame = ctk.CTkFrame(self)
        btm_frame.pack(pady=5)

        ctk.CTkButton(btm_frame, text="⬅ Volver", command=self.volver, width=150,
                      fg_color="#6a5acd", hover_color="#483d8b").grid(row=0, column=0, padx=10)

        ctk.CTkButton(btm_frame, text="💾 Guardar Resultado", command=self.save_results, width=180,
                      fg_color="#6a5acd", hover_color="#483d8b").grid(row=0, column=1, padx=10)

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
        while self.camera_active:
            ret, frame = self.cap.read()
            if not ret:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb).resize((640, 400))
            imgtk = ImageTk.PhotoImage(img)
            self.image_canvas.configure(image=imgtk)
            self.image_canvas.image = imgtk

    def capturar_foto(self):
        if self.camera_active and self.cap:
            ret, frame = self.cap.read()
            if ret:
                self.camera_active = False
                self.cap.release()
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(rgb).resize((640, 400))
                imgtk = ImageTk.PhotoImage(img)
                self.image_canvas.configure(image=imgtk)
                self.image_canvas.image = imgtk
                self.result_label.configure(text="Analizando emoción...")
                threading.Thread(target=self.analyze_emotion_thread, args=(frame.copy(),), daemon=True).start()
            else:
                self.result_label.configure(text="No se pudo capturar imagen.")
        else:
            self.result_label.configure(text="Cámara no activa.")

    def analyze_emotion_thread(self, frame):
        try:
            result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
            dominant = result[0]['dominant_emotion']
            if dominant.lower() in self.emotion_translations:
                translated = self.emotion_translations[dominant.lower()]
                self.result_label.configure(text=f"Emoción detectada: {translated}")

                with open("historial_emociones.txt", "a", encoding="utf-8") as f:
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    f.write(f"{timestamp} | {translated}\n")
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

class Historial(ctk.CTkFrame):
    def __init__(self, master):
        super().__init__(master)
        self.master = master

        ctk.CTkLabel(self, text="Historial de Emociones", font=("Arial", 24), text_color="#483D8B").pack(pady=10)

        self.scroll_frame = ctk.CTkScrollableFrame(self, width=800, height=500)
        self.scroll_frame.pack(pady=10)

        btn_frame = ctk.CTkFrame(self)
        btn_frame.pack(pady=10)

        ctk.CTkButton(btn_frame, text="🔄 Recargar", command=self.cargar_historial, width=150,
                      fg_color="#6a5acd", hover_color="#483d8b").grid(row=0, column=0, padx=10)

        ctk.CTkButton(btn_frame, text="🗑 Borrar Todo", command=self.borrar_historial, width=150,
                      fg_color="red", hover_color="#8B0000").grid(row=0, column=1, padx=10)

        ctk.CTkButton(btn_frame, text="⬅ Volver", command=self.volver, width=150,
                      fg_color="#6a5acd", hover_color="#483d8b").grid(row=0, column=2, padx=10)

        self.cargar_historial()

    def cargar_historial(self):
        for widget in self.scroll_frame.winfo_children():
            widget.destroy()

        try:
            with open("historial_emociones.txt", "r", encoding="utf-8") as f:
                lineas = f.readlines()
                if not lineas:
                    ctk.CTkLabel(self.scroll_frame, text="Sin datos aún.").pack(pady=10)
                    return
                for linea in lineas:
                    ctk.CTkLabel(self.scroll_frame, text=linea.strip(), anchor="w").pack(padx=10, pady=4, fill="x")
        except FileNotFoundError:
            ctk.CTkLabel(self.scroll_frame, text="No se encontró historial.").pack(pady=10)

    def borrar_historial(self):
        open("historial_emociones.txt", "w").close()
        self.cargar_historial()

    def volver(self):
        self.master.mostrar_ventana(Inicio)

class EmotionSenseApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("EmotionSense AI")
        ancho, alto = 950, 720
        x = (self.winfo_screenwidth() // 2) - (ancho // 2)
        y = (self.winfo_screenheight() // 2) - (alto // 2)
        self.geometry(f"{ancho}x{alto}+{x}+{y}")
        self.resizable(False, False)

        self.frames = {}
        for F in (Inicio, Analisis, Historial):
            frame = F(self)
            self.frames[F] = frame
            frame.place(relwidth=1, relheight=1)

        self.mostrar_ventana(Inicio)

    def mostrar_ventana(self, contenedor):
        self.frames[contenedor].tkraise()

if __name__ == "__main__":
    app = EmotionSenseApp()
    app.mainloop()