import customtkinter as ctk
import tkinter as tk

class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("EmotionSense AI")
        self.geometry("600x400")
        self.resizable(False, False)

        # Contenedor de todas las ventanas
        self.frames = {}

        for F in (Inicio, Analisis):
            frame = F(self)
            self.frames[F] = frame
            frame.place(relwidth=1, relheight=1)

        self.mostrar_ventana(Inicio)

    def mostrar_ventana(self, contenedor):
        frame = self.frames[contenedor]
        frame.tkraise()


class Inicio(ctk.CTkFrame):
    def __init__(self, master):
        super().__init__(master)
        self.master = master

        ctk.CTkLabel(self, text="Bienvenido a EmotionSense AI", font=("Arial", 24)).pack(pady=30)
        ctk.CTkButton(self, text="Iniciar Análisis de Emociones", command=self.ir_a_analisis).pack(pady=10)
        ctk.CTkButton(self, text="Salir", command=self.master.quit).pack(pady=10)

    def ir_a_analisis(self):
        self.master.mostrar_ventana(Analisis)


class Analisis(ctk.CTkFrame):
    def __init__(self, master):
        super().__init__(master)
        self.master = master

        ctk.CTkLabel(self, text="Análisis Facial", font=("Arial", 22)).pack(pady=20)
        ctk.CTkButton(self, text="Regresar al Inicio", command=self.ir_a_inicio).pack(pady=10)
        # Aquí más adelante se agregará la lógica de la cámara y análisis de emociones

    def ir_a_inicio(self):
        self.master.mostrar_ventana(Inicio)


if __name__ == "__main__":
    ctk.set_appearance_mode("dark")  # o "light"
    ctk.set_default_color_theme("blue")  # Puedes cambiarlo a "green", "dark-blue", etc.
    app = App()
    app.mainloop()

