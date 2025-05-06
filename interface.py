import tkinter as tk
from tkinter import ttk, filedialog
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import control as ctrl
import numpy as np
from scipy.io import loadmat, whosmat

# Função para carregar dados do arquivo .mat
def load_dados(filename):
    data = loadmat(filename)
    data = data.get(whosmat(filename)[0][0])[0][0]
    Tempo, Entrada, Saida, QuantidadeFisica, Unidades = data
    return Tempo[0], Entrada[0], Saida[0], QuantidadeFisica[0], Unidades[0]

# Interface principal de visualização de dados e identificação
class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Visualizador de Sistema")

        # Botões
        tk.Button(root, text="Carregar Dados", command=self.carregar_dados).pack(pady=5)
        tk.Button(root, text="Plotar Entrada/Saída", command=self.plot_entrada_saida).pack(pady=5)
        tk.Button(root, text="Identificação Smith", command=self.plot_identificacao_smith).pack(pady=5)
        tk.Button(root, text="Identificação Sundaresan", command=self.plot_identificacao_sundaresan).pack(pady=5)
        tk.Button(root, text="Abrir Simulação PID", command=self.abrir_pid_interface).pack(pady=5)

        # Área para o gráfico
        self.fig, self.ax = plt.subplots(figsize=(6, 4))
        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas.get_tk_widget().pack(padx=10, pady=10)

        self.tempo = self.entrada = self.saida = None

    def carregar_dados(self):
        path = filedialog.askopenfilename(filetypes=[("Arquivos .mat", "*.mat")])
        if path:
            self.tempo, self.entrada, self.saida, *_ = load_dados(path)
            self.ax.clear()
            self.ax.set_title("Dados carregados com sucesso!")
            self.canvas.draw()

    def plot_entrada_saida(self):
        if self.tempo is None:
            return
        self.ax.clear()
        self.ax.plot(self.tempo, self.entrada, label="Entrada")
        self.ax.plot(self.tempo, self.saida, label="Saída", linestyle='--')
        self.ax.set_title("Entrada e Saída")
        self.ax.set_xlabel("Tempo [s]")
        self.ax.set_ylabel("Amplitude")
        self.ax.grid(True)
        self.ax.legend()
        self.canvas.draw()

    def plot_identificacao_smith(self):
        if self.tempo is None:
            return
        saida_ini = np.min(self.saida)
        saida_fim = np.max(self.saida)
        delta_saida = saida_fim - saida_ini
        delta_entrada = self.entrada.mean()
        k = delta_saida / delta_entrada
        y283 = saida_ini + 0.283 * delta_saida
        y632 = saida_ini + 0.632 * delta_saida
        t1 = np.interp(y283, self.saida, self.tempo)
        t2 = np.interp(y632, self.saida, self.tempo)
        tau = 1.5 * (t2 - t1)
        theta = t1 - 0.3 * tau

        self.ax.clear()
        self.ax.plot(self.tempo, self.saida, label="Saída real")
        self.ax.axhline(y=y283, color="green", linestyle="--", label="28.3%")
        self.ax.axhline(y=y632, color="purple", linestyle="--", label="63.2%")
        self.ax.axvline(x=t1, color="green")
        self.ax.axvline(x=t2, color="purple")
        self.ax.set_title(f"Identificação via Smith (K={k:.2f}, Tau={tau:.2f}, Theta={theta:.2f})")
        self.ax.set_xlabel("Tempo [s]")
        self.ax.set_ylabel("Saída")
        self.ax.grid(True)
        self.ax.legend()
        self.canvas.draw()

    def plot_identificacao_sundaresan(self):
        if self.tempo is None:
            return
        entrada_i = np.mean(self.entrada[:10])
        entrada_f = np.mean(self.entrada[-10:])
        saida_i = np.mean(self.saida[:10])
        saida_f = np.mean(self.saida[-10:])
        delta_entrada = entrada_f - entrada_i
        delta_saida = saida_f - saida_i
        k = delta_saida / self.entrada.mean()
        saida_norm = (self.saida - saida_i) / delta_saida

        y1, y2 = 0.353, 0.853
        t1 = self.tempo[np.where(saida_norm >= y1)[0][0]]
        t2 = self.tempo[np.where(saida_norm >= y2)[0][0]]
        tau = 0.67 * (t2 - t1)
        theta = 1.3 * t1 - 0.29 * t2

        self.ax.clear()
        self.ax.plot(self.tempo, self.saida, label="Saída real")
        self.ax.axvline(t1, color='red', linestyle='--', label="t1 (35.3%)")
        self.ax.axvline(t2, color='blue', linestyle='--', label="t2 (85.3%)")
        self.ax.set_title(f"Identificação via Sundaresan (K={k:.2f}, Tau={tau:.2f}, Theta={theta:.2f})")
        self.ax.set_xlabel("Tempo [s]")
        self.ax.set_ylabel("Saída")
        self.ax.grid(True)
        self.ax.legend()
        self.canvas.draw()

    def abrir_pid_interface(self):
        nova_janela = tk.Toplevel(self.root)
        PIDInterfaceGrupo3(nova_janela)

# Interface para simulação PID
class PIDInterfaceGrupo3:
    def __init__(self, root):
        self.root = root
        self.root.title("Controle PID - Projeto C213 Grupo 3")

        self.metodo = tk.StringVar(value="CHR com sobrevalor")
        self.vars = {}

        frame_top = tk.LabelFrame(self.root, text="Configuração de Sintonia")
        frame_top.pack(pady=5, fill="x", padx=10)
        ttk.Label(frame_top, text="Seleção de Sintonia:").pack(side=tk.LEFT, padx=5)
        combo = ttk.Combobox(frame_top, textvariable=self.metodo,
                             values=["CHR com sobrevalor", "Cohen-Coon"], state="readonly", width=20)
        combo.pack(side=tk.LEFT, padx=5)
        combo.bind("<<ComboboxSelected>>", self.atualizar_campos)

        frame_pid = tk.LabelFrame(self.root, text="Parâmetros PID")
        frame_pid.pack(pady=5, fill="x", padx=10)
        for label in ["Kp", "Ti", "Td", "SetPoint"]:
            tk.Label(frame_pid, text=label + ":").grid(row=len(self.vars), column=0, sticky='e')
            var = tk.StringVar(value="1.0")
            entry = tk.Entry(frame_pid, textvariable=var, width=10)
            entry.grid(row=len(self.vars), column=1, padx=5, pady=2)
            self.vars[label] = (var, entry)

        frame_planta = tk.LabelFrame(self.root, text="Parâmetros da Planta (K, τ, θ)")
        frame_planta.pack(pady=5, fill="x", padx=10)
        for label in ["K", "Tau", "Theta"]:
            tk.Label(frame_planta, text=label + ":").grid(row=len(self.vars), column=0, sticky='e')
            var = tk.StringVar(value="1.0")
            entry = tk.Entry(frame_planta, textvariable=var, width=10)
            entry.grid(row=len(self.vars), column=1, padx=5, pady=2)
            self.vars[label] = (var, entry)

        frame_btn = tk.Frame(self.root)
        frame_btn.pack(pady=10)
        tk.Button(frame_btn, text="Simular", command=self.simular).pack(side=tk.LEFT, padx=10)
        tk.Button(frame_btn, text="Sair", command=self.root.destroy).pack(side=tk.LEFT)

        self.fig, self.ax = plt.subplots(figsize=(5, 3))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack()
        self.atualizar_campos()

    def atualizar_campos(self, event=None):
        metodo = self.metodo.get()
        presets = {
            "CHR com sobrevalor": {"Kp": 3.96, "Ti": 37.0, "Td": 9.25, "SetPoint": 45},
            "Cohen-Coon": {"Kp": 2.58, "Ti": 85.0, "Td": 14.3, "SetPoint": 45}
        }
        for key in ["Kp", "Ti", "Td", "SetPoint"]:
            self.vars[key][0].set(str(presets[metodo][key]))
        self.vars["K"][0].set("5006.78")
        self.vars["Tau"][0].set("3315")
        self.vars["Theta"][0].set("1580.5")

    def simular(self):
        try:
            Kp = float(self.vars["Kp"][0].get())
            Ti = float(self.vars["Ti"][0].get())
            Td = float(self.vars["Td"][0].get())
            SP = float(self.vars["SetPoint"][0].get())
            k = float(self.vars["K"][0].get())
            tau = float(self.vars["Tau"][0].get())
            theta = float(self.vars["Theta"][0].get())
        except ValueError:
            self.ax.clear()
            self.ax.set_title("Parâmetros inválidos.")
            self.canvas.draw()
            return

        G = ctrl.tf([k], [tau, 1])
        PID = ctrl.tf([Kp * Td, Kp, Kp / Ti], [1, 0])
        Gmf = ctrl.feedback(PID * G, 1)
        T = np.linspace(0, 3500, 1000)
        t_sim, y_sim = ctrl.step_response(Gmf, T)
        atraso_amostras = np.searchsorted(T, theta)
        y_atrasada = np.concatenate((np.zeros(atraso_amostras), y_sim))[:len(T)]
        y_out = y_atrasada * SP

        self.ax.clear()
        self.ax.plot(T, y_out, label=f"{self.metodo.get()} (K={k}, τ={tau}, θ={theta})")
        self.ax.set_title("Resposta do Sistema com PID")
        self.ax.set_xlabel("Tempo [s]")
        self.ax.set_ylabel("Saída")
        self.ax.grid(True)
        self.ax.legend()
        self.canvas.draw()

if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()