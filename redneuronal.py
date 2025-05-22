import customtkinter as ctk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
from deepface import DeepFace
import threading
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import time


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

        ctk.CTkButton(self, text="📊 Ver Gráfica", command=self.ir_a_historial, width=200,
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
               "angry": "Enojado",
               "disgust": "Disgustado",
               "fear": "Con miedo",
               "happy": "Feliz",
               "sad": "Triste",
               "surprise": "Sorprendido",
               "neutral": "Neutral"
        }

        ctk.CTkLabel(self, text="Análisis Facial", font=("Segoe UI Black", 22), text_color='#483D8B').pack(pady=10)

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
                      fg_color="#a20000", hover_color="#720000", border_color="black").grid(row=0, column=1, padx=10)

        ctk.CTkButton(btm_frame, text="💾 Guardar Resultado", command=self.save_results, width=180,
                       fg_color="#3ba200", hover_color="#297200", border_color="black").grid(row=0, column=0, padx=10)

    def volver(self):
        if self.camera_active:
            self.camera_active = False
            self.cap.release()
        self.master.mostrar_ventana(Inicio)
    
    def toggle_camera(self):
        if self.camera_active:
            self.camera_active = False
            self.cap.release()
            self.image_canvas.configure(image=None, text="Activa tu camara")
        else:
            self.cap = cv2.VideoCapture(0)
            self.camera_active = True
            threading.Thread(target=self.show_camera, daemon=True).start()

    def show_camera(self):
        while self.camera_active:
            ret, frame = self.cap.read()
            if not ret:
                break

            analisis = None
            try:
                analisis = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
            except Exception as e:
                print("Error analizando emociones:", e)
                self.reiniciar_camara_con_retraso()
                break  # Detiene el bucle show_camera si hay un error

            if analisis:
                emocion = analisis[0].get("dominant_emotion", "Emoción no válida")

                if emocion in ["Emoción no válida", "Error detectando emoción"]:
                    print("Emoción no válida detectada. Reiniciando cámara...")
                    self.reiniciar_camara_con_retraso()
                    break  # Salir del bucle para permitir reinicio

                # Acceder a las coordenadas del rostro
                x = analisis[0]['region']['x']
                y = analisis[0]['region']['y']
                w = analisis[0]['region']['w']
                h = analisis[0]['region']['h']
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb).resize((640, 400))
            imgtk = ImageTk.PhotoImage(img)
            self.image_canvas.configure(image=imgtk)
            self.image_canvas.image = imgtk
            translated = self.emotion_translations.get(emocion.lower(), emocion.capitalize())
            self.result_label.configure(text=f"Emoción detectada: {translated}")
            self.emocion_actual = translated  # Guarda la emoción actual para luego guardarla


    def reiniciar_camara_con_retraso(self):
        def esperar_y_reactivar():
            time.sleep(3)
            self.toggle_camera()  # Reactiva la cámara correctamente

        self.toggle_camera()  # Desactiva la cámara
        threading.Thread(target=esperar_y_reactivar, daemon=True).start()

    def capturar_foto(self):
        if self.camera_active and self.cap:
            ret, frame = self.cap.read()
            if ret:
                self.camera_active = False
                self.cap.release()
                frame = cv2.flip(frame, 1)
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
                self.reiniciar_camara_con_retraso()
        except Exception:
            self.result_label.configure(text="Error detectando emoción")
            self.reiniciar_camara_con_retraso()
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

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class Historial(ctk.CTkFrame):
    def __init__(self, master):
        super().__init__(master)
        self.master = master

        ctk.CTkLabel(self, text="Total de Emociones", font=("Segoe UI Black", 24), text_color="#483D8B").pack(pady=10)

        self.scroll_frame = ctk.CTkScrollableFrame(self, width=800, height=500)
        self.scroll_frame.pack(pady=10)

        # Crear un frame donde se dibujará la gráfica
        self.graph_frame = ctk.CTkFrame(self.scroll_frame, width=800, height=500)
        self.graph_frame.pack(padx=10, pady=10)

        btn_frame = ctk.CTkFrame(self)
        btn_frame.pack(pady=10)

        ctk.CTkButton(btn_frame, text="🔄 Actualizar Gráfica", command=self.cargar_grafica, width=150,
                      fg_color="#3ba200", hover_color="#297200", border_color="black").grid(row=0, column=0, padx=10)

        ctk.CTkButton(btn_frame, text="💾 Guardar Gráfica", command=self.guardar_grafica, width=150,
                      fg_color="#6a5acd", hover_color="#483d8b", border_color="black").grid(row=0, column=1, padx=10)

        ctk.CTkButton(btn_frame, text="⬅ Volver", command=self.volver, width=150,
                      fg_color="#a20000", hover_color="#720000", border_color="black").grid(row=0, column=2, padx=10)

        self.figure = None  # Para almacenar la figura actual
        self.cargar_grafica()

    def cargar_grafica(self):
        # Limpiar la gráfica anterior si existe
        for widget in self.graph_frame.winfo_children():
            widget.destroy()

        # Cargar las emociones desde el archivo de historial
        emociones = {'Enojo': 0, 'Feliz': 0, 'Triste': 0}
        try:
            with open("historial_emociones.txt", "r", encoding="utf-8") as f:
                lineas = f.readlines()
                for linea in lineas:
                    emocion = linea.strip().split(" | ")[1]
                    if emocion in emociones:
                        emociones[emocion] += 1
        except FileNotFoundError:
            ctk.CTkLabel(self.scroll_frame, text="No se encontró historial.").pack(pady=10)
            return

        # Crear la gráfica
        self.figure, ax = plt.subplots(figsize=(9, 7))
        ax.bar(emociones.keys(), emociones.values(), color=['red', 'yellow', 'blue'])
        ax.set_xlabel("Emociones")
        ax.set_ylabel("Frecuencia")
        ax.set_title("Frecuencia de Emociones Detectadas")

        # Mostrar la gráfica en el frame de la interfaz
        canvas = FigureCanvasTkAgg(self.figure, master=self.graph_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)

    def guardar_grafica(self):
        if self.figure:
            archivo = filedialog.asksaveasfilename(defaultextension=".png",
                                                   filetypes=[("PNG Image", "*.png")],
                                                   title="Guardar gráfica como...")
            if archivo:
                self.figure.savefig(archivo)

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