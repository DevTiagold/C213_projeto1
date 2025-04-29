import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import control as ctrl
import numpy as np

class PIDInterfaceGrupo3:
    def __init__(self, root):
        self.root = root
        self.root.title("Controle PID - Projeto C213 Grupo 3")

        self.metodo = tk.StringVar(value="CHR com sobrevalor")

        
        frame_top = tk.Frame(self.root)
        frame_top.pack(pady=10)

        ttk.Label(frame_top, text="Seleção de Sintonia:").pack(side=tk.LEFT)
        combo = ttk.Combobox(frame_top, textvariable=self.metodo, values=["CHR com sobrevalor", "Cohen-Coon"], state="readonly", width=20)
        combo.pack(side=tk.LEFT)
        combo.bind("<<ComboboxSelected>>", self.atualizar_campos)

       
        self.vars = {}
        frame_pid = tk.Frame(self.root)
        frame_pid.pack(pady=10)

        for label in ["Kp", "Ti", "Td", "SetPoint"]:
            tk.Label(frame_pid, text=label + ":").grid(row=len(self.vars), column=0, sticky='e')
            var = tk.StringVar(value="1.0")
            entry = tk.Entry(frame_pid, textvariable=var, width=10)
            entry.grid(row=len(self.vars), column=1, padx=5, pady=2)
            self.vars[label] = (var, entry)

       
        frame_btn = tk.Frame(self.root)
        frame_btn.pack(pady=10)

        tk.Button(frame_btn, text="Simular", command=self.simular).pack(side=tk.LEFT, padx=10)
        tk.Button(frame_btn, text="Sair", command=root.quit).pack(side=tk.LEFT)

    
        self.fig, self.ax = plt.subplots(figsize=(5, 3))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack()

        self.atualizar_campos()

    def atualizar_campos(self, event=None):
        metodo = self.metodo.get()
        # Para essas técnicas, todos os campos são editáveis, então nada é desativado
        for _, entry in self.vars.values():
            entry.config(state="normal")

        if metodo == "CHR com sobrevalor":
            self.vars["Kp"][0].set("3.96")
            self.vars["Ti"][0].set("37.0")
            self.vars["Td"][0].set("9.25")
            self.vars["SetPoint"][0].set("45")

        elif metodo == "Cohen-Coon":
            self.vars["Kp"][0].set("2.58")
            self.vars["Ti"][0].set("85.0")
            self.vars["Td"][0].set("14.3")
            self.vars["SetPoint"][0].set("45")

    def simular(self):
        try:
            Kp = float(self.vars["Kp"][0].get())
            Ti = float(self.vars["Ti"][0].get())
            Td = float(self.vars["Td"][0].get())
            SP = float(self.vars["SetPoint"][0].get())
        except ValueError:
            print(" Parâmetros inválidos.")
            return

        # Planta simulada 
        k, tau, theta = 5006.78, 3315, 1580.5
        G = ctrl.tf([k], [tau, 1])
        PID = ctrl.tf([Kp * Td, Kp, Kp / Ti], [1, 0])
        Gmf = ctrl.feedback(PID * G, 1)

        T = np.linspace(0, 3500, 1000)
        t_sim, y_sim = ctrl.step_response(Gmf, T)

        atraso_amostras = np.searchsorted(T, theta)
        y_atrasada = np.concatenate((np.zeros(atraso_amostras), y_sim))[:len(T)]
        y_out = y_atrasada * SP

        self.ax.clear()
        self.ax.plot(T, y_out, label=self.metodo.get())
        self.ax.set_title("Resposta do Sistema com PID")
        self.ax.set_xlabel("Tempo [s]")
        self.ax.set_ylabel("Saída")
        self.ax.grid(True)
        self.ax.legend()
        self.canvas.draw()

if __name__ == "__main__":
    root = tk.Tk()
    app = PIDInterfaceGrupo3(root)
    root.mainloop()
