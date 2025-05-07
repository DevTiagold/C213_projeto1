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

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Identificação de Processos & Sintonia de Controladores PID")
        self.root.geometry("850x500")

        # Título principal
        titulo = tk.Label(root, text="Projeto Prático C213 - Sistemas Embarcados", font=("Segoe UI", 14, "bold"))
        titulo.pack(pady=(10, 0))

        subtitulo = tk.Label(root, text="Identificação de Processos & Sintonia de Controladores PID", font=("Segoe UI", 11))
        subtitulo.pack(pady=(0, 10))

        # Divisão em dois frames (lateral e conteúdo principal)
        frame_principal = tk.Frame(root)
        frame_principal.pack(fill="both", expand=True)

        # Frame lateral (controle)
        self.frame_lateral = ttk.Frame(frame_principal, padding=10)
        self.frame_lateral.pack(side="left", fill="y")

        self.botao_arquivo = ttk.Button(self.frame_lateral, text="Escolher Arquivo", command=self.carregar_dados)
        self.botao_arquivo.pack(pady=5, fill="x")

        self.label_alerta = tk.Label(self.frame_lateral, text="⚠️ Selecione um dataset.", fg="orange", font=("Segoe UI", 9, "italic"))
        self.label_alerta.pack(pady=5)

        ttk.Separator(self.frame_lateral, orient="horizontal").pack(fill="x", pady=10)

        ttk.Button(self.frame_lateral, text="Entrada/Saída", command=self.plot_entrada_saida).pack(fill="x", pady=3)
        ttk.Button(self.frame_lateral, text="Identificação Smith", command=self.plot_identificacao_smith).pack(fill="x", pady=3)
        ttk.Button(self.frame_lateral, text="Identificação Sundaresan", command=self.plot_identificacao_sundaresan).pack(fill="x", pady=3)
        ttk.Button(self.frame_lateral, text="Abrir Simulação PID", command=self.abrir_pid_interface).pack(fill="x", pady=3)

        # Campos de exibição dos parâmetros
        self.param_frame = ttk.LabelFrame(self.frame_lateral, text="Identificação")
        self.param_frame.pack(pady=10, fill="x")

        self.k_label = ttk.Label(self.param_frame, text="kₚ: -")
        self.k_label.pack(anchor="w")
        self.tau_label = ttk.Label(self.param_frame, text="τ: -")
        self.tau_label.pack(anchor="w")
        self.theta_label = ttk.Label(self.param_frame, text="θ: -")
        self.theta_label.pack(anchor="w")
        self.erro_label = ttk.Label(self.param_frame, text="Eₘ: -")
        self.erro_label.pack(anchor="w")

        # Frame gráfico principal
        frame_grafico = tk.Frame(frame_principal)
        frame_grafico.pack(side="right", fill="both", expand=True)

        self.fig, self.ax = plt.subplots(figsize=(6, 4))
        self.canvas = FigureCanvasTkAgg(self.fig, master=frame_grafico)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        # Inicialização de variáveis
        self.tempo = self.entrada = self.saida = None
        self.k_id = self.tau_id = self.theta_id = None

    def carregar_dados(self):
        path = filedialog.askopenfilename(filetypes=[("Arquivos .mat", "*.mat")])
        if path:
            self.tempo, self.entrada, self.saida, *_ = load_dados(path)
            self.ax.clear()
            self.ax.set_title("Dados carregados com sucesso!")
            self.canvas.draw()
            self.label_alerta.config(text="✔️ Dataset carregado.", fg="green")

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
        saida_ini = self.saida[0]
        saida_fim = self.saida[-1]
        delta_saida = saida_fim - saida_ini
        delta_entrada = self.entrada.mean()
        k = delta_saida / delta_entrada
        
        y1 = saida_ini + 0.283 * delta_saida
        y2 = saida_ini + 0.632 * delta_saida

        t1 = self.tempo[np.where(self.saida >= y1)[0][0]]
        t2 = self.tempo[np.where(self.saida >= y2)[0][0]]

        tau = 1.5 * (t2 - t1)
        theta = t2 - tau

        self.k_id, self.tau_id, self.theta_id = k, tau, theta
        self.k_label.config(text=f"kₚ: {k:.2f}")
        self.tau_label.config(text=f"τ: {tau:.2f}")
        self.theta_label.config(text=f"θ: {theta:.2f}")
        self.erro_label.config(text="Eₘ: -")

        self.ax.clear()
        self.ax.plot(self.tempo, self.saida, label="Saída real")
        self.ax.axhline(y=y1, color="green", linestyle="--", label="28.3%")
        self.ax.axhline(y=y2, color="purple", linestyle="--", label="63.2%")
        self.ax.axvline(x=t1, color="green")
        self.ax.axvline(x=t2, color="purple")
        self.ax.set_title(f"Identificação Smith")
        self.ax.set_xlabel("Tempo [s]")
        self.ax.set_ylabel("Saída")
        self.ax.grid(True)
        self.ax.legend()
        self.canvas.draw()

    def plot_identificacao_sundaresan(self):
        if self.tempo is None:
            return
        #entrada_i = np.mean(self.entrada[:10])
        #entrada_f = np.mean(self.entrada[-10:])
        saida_ini = np.mean(self.saida[:10])
        saida_fim = np.mean(self.saida[-10:])
        delta_entrada = self.entrada.mean()
        delta_saida = saida_fim - saida_ini
        
        k = delta_saida/ delta_entrada
        #k = delta_saida / self.entrada.mean()
        saida_norm = (self.saida - saida_ini) / delta_saida

        y1, y2 = 0.353, 0.853
        t1 = self.tempo[np.where(saida_norm >= y1)[0][0]]
        t2 = self.tempo[np.where(saida_norm >= y2)[0][0]]
        tau = 0.67 * (t2 - t1)
        theta = 1.3 * t1 - 0.29 * t2

        self.k_id, self.tau_id, self.theta_id = k, tau, theta
        self.k_label.config(text=f"kₚ: {k:.2f}")
        self.tau_label.config(text=f"τ: {tau:.2f}")
        self.theta_label.config(text=f"θ: {theta:.2f}")
        self.erro_label.config(text="Eₘ: -")

        self.ax.clear()
        self.ax.plot(self.tempo, self.saida, label="Saída real")
        self.ax.axvline(t1, color='red', linestyle='--', label="t1 (35.3%)")
        self.ax.axvline(t2, color='blue', linestyle='--', label="t2 (85.3%)")
        self.ax.set_title("Identificação Sundaresan")
        self.ax.set_xlabel("Tempo [s]")
        self.ax.set_ylabel("Saída")
        self.ax.grid(True)
        self.ax.legend()
        self.canvas.draw()

        
    def abrir_pid_interface(self):
            nova_janela = tk.Toplevel(self.root)
            PIDInterfaceGrupo3(nova_janela, self.k_id, self.tau_id, self.theta_id)
    # Interface PID
class PIDInterfaceGrupo3:
    def __init__(self, root, k=None, tau=None, theta=None):
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
 
        self.k_default = k if k is not None else 5006.78
        self.tau_default = tau if tau is not None else 3315
        self.theta_default = theta if theta is not None else 1580.5
 
        self.atualizar_campos()
 
    def atualizar_campos(self, event=None):
        metodo = self.metodo.get()
        presets = {
            "CHR com sobrevalor": {"Kp": 3.96, "Ti": 37.0, "Td": 9.25, "SetPoint": 45},
            "Cohen-Coon": {"Kp": 2.58, "Ti": 85.0, "Td": 14.3, "SetPoint": 45}
        }
        for key in ["Kp", "Ti", "Td", "SetPoint"]:
            self.vars[key][0].set(str(presets[metodo][key]))
        self.vars["K"][0].set(str(self.k_default))
        self.vars["Tau"][0].set(str(self.tau_default))
        self.vars["Theta"][0].set(str(self.theta_default))
 
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
        self.ax.axhline(SP, color='gray', linestyle='--', label='Setpoint')
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
