from scipy.io import loadmat, whosmat
import numpy as np
import matplotlib.pyplot as plt
#from matplotlib.widgets import Slider, Button # usar quando for implementar interface
import control as ctrl



def load_dados(filename):
    data = loadmat(filename)
    data = data.get(whosmat(filename)[0][0])[0][0]
    Tempo, Entrada, Saida, QuantidadeFisica, Unidades = data
    return Tempo[0], Entrada[0], Saida[0], QuantidadeFisica[0], Unidades[0]

def plot_entrada_saida(Tempo, Entrada, Saida):
    plt.plot(Tempo, Entrada, label='Entrada')
    plt.plot(Tempo, Saida, label='Saída')
    plt.title('Entrada vs Saída')
    plt.xlabel('Tempo [s]')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_entrada_saida_aprimorada(tempo, entrada, saida):
    plt.figure(figsize=(10, 5))
    plt.plot(tempo, entrada, label='Entrada')
    plt.plot(tempo, saida, label='Saída', linestyle='--')
    plt.title('Curva de Entrada e Saída do Sistema')
    plt.xlabel('Tempo [s]')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.legend()
    plt.xlim(0, 2500)  # mais q o tempo disponível do dataset
    plt.tight_layout()
    plt.show()

def identificar_planta_smith(tempo, entrada, saida):
    entrada_inicial = np.mean(entrada[:10])
    entrada_final = np.mean(entrada[-10:])
    saida_inicial = np.mean(saida[:10])
    saida_final = np.mean(saida[-10:])

    delta_entrada = entrada_final - entrada_inicial
    delta_saida = saida_final - saida_inicial
    k = delta_saida / delta_entrada

    saida_normalizada = (saida - saida_inicial) / delta_saida

    # 28.3% e 63.2% são pontos padrões do método de Smith
    y_283 = 0.283
    y_632 = 0.632

    t1 = tempo[np.where(saida_normalizada >= y_283)[0][0]]
    t2 = tempo[np.where(saida_normalizada >= y_632)[0][0]]

    tau = 1.5 * (t2 - t1)
    theta = t1 - 0.3 * tau

    print(f"\n Identificação via Método de Smith:")
    print(f"Ganho (k): {k:.4f}")
    print(f"Constante de tempo (tau): {tau:.2f} s")
    print(f"Atraso (theta): {theta:.2f} s")

    return k, tau, theta

def identificar_planta_sundaresan(tempo, entrada, saida):
    entrada_inicial = np.mean(entrada[:10])
    entrada_final = np.mean(entrada[-10:])
    saida_inicial = np.mean(saida[:10])
    saida_final = np.mean(saida[-10:])

    delta_entrada = entrada_final - entrada_inicial
    delta_saida = saida_final - saida_inicial
    k = delta_saida / delta_entrada

    saida_norm = (saida - saida_inicial) / delta_saida

    # Padrões do método 
    y1, y2 = 0.353, 0.853

    t1 = tempo[np.where(saida_norm >= y1)[0][0]]
    t2 = tempo[np.where(saida_norm >= y2)[0][0]]

    tau = 0.67 * (t2 - t1)
    theta = 1.3 * t1 - 0.29 * t2

    print(f"\n Identificação via Método de Sundaresan:")
    print(f"Ganho (k): {k:.4f}")
    print(f"Constante de tempo (tau): {tau:.2f} s")
    print(f"Atraso (theta): {theta:.2f} s")

    return k, tau, theta


def obter_funcao_transferencia_pade(k, tau, theta, ordem=20):
    G_sem_atraso = ctrl.tf([k], [tau, 1])
    num_pade, den_pade = ctrl.pade(theta, ordem)
    atraso_pade = ctrl.tf(num_pade, den_pade)
    G_pade = ctrl.series(G_sem_atraso, atraso_pade)
    return G_pade


# Modelar TF
def obter_funcao_transferencia(k, tau, theta):
    #G = ctrl.tf([k], [tau, 1]) # sem pade
    G = obter_funcao_transferencia_pade(k, tau, theta, ordem=20)
    return G, theta


def comparar_pade_vs_padding(k, tau, theta, tempo_max=4000):
    T = np.linspace(0, tempo_max, 1000)

    # Atraso via padding --> zero 
    G_padding = ctrl.tf([k], [tau, 1])
    t_padding, y_padding = ctrl.step_response(G_padding, T)
    atraso_samples = np.searchsorted(T, theta)
    y_padding_atrasado = np.concatenate((np.zeros(atraso_samples), y_padding))[:len(T)]

    # Atraso via Padé 
    num_pade, den_pade = ctrl.pade(theta, 20)
    atraso_pade = ctrl.tf(num_pade, den_pade)
    G_sem_atraso = ctrl.tf([k], [tau, 1])
    G_pade = ctrl.series(G_sem_atraso, atraso_pade)
    t_pade, y_pade = ctrl.step_response(G_pade, T)

    # Gráfico comparativo pade/padding
    plt.figure(figsize=(10,6))
    plt.plot(t_padding, y_padding_atrasado, label='Modelo com Zero-Padding (Simples)', linestyle='--')
    plt.plot(t_pade, y_pade, label='Modelo com Aproximação de Padé (Ordem 20)', linestyle='-')
    plt.title('Comparação entre Aproximação de Atraso: Zero-Padding vs Padé (20ª Ordem)')
    plt.xlabel('Tempo [s]')
    plt.ylabel('Saída')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def comparar_pade_ordens(k, tau, theta, tempo_max=4000):
    T = np.linspace(0, tempo_max, 1000)
    G_base = ctrl.tf([k], [tau, 1])

    # Padé ordem 5
    num5, den5 = ctrl.pade(theta, 5)
    G_pade5 = ctrl.series(G_base, ctrl.tf(num5, den5))
    t5, y5 = ctrl.step_response(G_pade5, T)

    # Padé ordem 20
    num20, den20 = ctrl.pade(theta, 20)
    G_pade20 = ctrl.series(G_base, ctrl.tf(num20, den20))
    t20, y20 = ctrl.step_response(G_pade20, T)

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(t5, y5, '--', label='Padé (ordem 5)', color='tab:orange')
    plt.plot(t20, y20, '-', label='Padé (ordem 20)', color='tab:blue')
    plt.title("Comparação: Padé Ordem 5 vs Ordem 20")
    plt.xlabel("Tempo [s]")
    plt.ylabel("Saída")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


# Comparação 
def comparar_modelo_saida(tempo, saida_real, G, theta):
    t_sim, y_sim = ctrl.step_response(G, T=tempo)

    # Aplicar o atraso --->> theta na resposta simulada
    atraso_segundos = theta  # segundos
    atraso_amostras = np.searchsorted(tempo, tempo[0] + atraso_segundos)

    # criar vetor deslocado
    y_sim_atrasado = np.concatenate((np.zeros(atraso_amostras), y_sim))[:len(tempo)]

    # plocar gráfico
    plt.figure(figsize=(10,5))
    plt.plot(tempo, saida_real / np.max(saida_real), label='Saída Real (normalizada)')
    plt.plot(tempo, y_sim_atrasado, '--', label=f'Modelo 1ª ordem (Smith)')
    plt.xlabel('Tempo [s]')
    plt.ylabel('Saída (normalizada)')
    plt.title('Comparação da resposta: Real vs Modelo de Smith')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # erro
    erro = np.mean(np.abs((saida_real / np.max(saida_real)) - y_sim_atrasado))
    print(f"\n Erro médio entre o modelo e a saída real: {erro:.4f}")

# Sintonizar PID 
# CHR com Sobressinal e Cohen-Coon

def sintonia_pid_chr_sobressinal(k, tau, theta):
    Kp = 0.6 * (tau / (k * theta))
    Ti = tau
    Td = 0.5 * theta
    return Kp, Ti, Td

def sintonia_pid_cohen_coon(k, tau, theta):
    R = theta / tau
    Kp = (1 / k) * (1.35 + (0.27 * R))
    Ti = tau * ((2.5 + 0.66 * R) / (1 + 0.444 * R))
    Td = tau * ((0.37 * R) / (1 + 0.444 * R))
    return Kp, Ti, Td

# Simulação PID

def simular_pid(tempo, G, theta, Kp, Ti, Td):
    PID = ctrl.tf([Kp * Td, Kp, Kp / Ti], [1, 0])
    Gmf = ctrl.feedback(PID * G, 1)

    t_out, y_out = ctrl.step_response(Gmf, T=tempo)

    # aplicar atraso
    atraso_amostras = np.searchsorted(tempo, tempo[0] + theta)
    y_out_atrasado = np.concatenate((np.zeros(atraso_amostras), y_out))[:len(tempo)]

    return t_out, y_out_atrasado

# Avaliar desempenho

def avaliar_desempenho(tempo, resposta):
    valor_final = np.mean(resposta[-10:])
    overshoot = ((np.max(resposta) - valor_final) / valor_final) * 100

    tolerancia = 0.02 * valor_final
    tempo_acomodacao = next((tempo[i] for i in range(len(resposta)) if all(abs(resposta[i:] - valor_final) < tolerancia)), None)

    print(f"Overshoot: {overshoot:.2f}%")
    print(f"Tempo de acomodação (±2%): {tempo_acomodacao:.2f} s")


# Para executar
if __name__ == "__main__":
    # Carregar dados
    Tempo, Entrada, Saida, QuantidadeFisica, Unidades = load_dados('Dataset_Grupo3.mat')

    Tempo = Tempo.astype(float)  

    plot_entrada_saida_aprimorada(Tempo, Entrada, Saida)
    
    # Identificação
    k, tau, theta = identificar_planta_smith(Tempo, Entrada, Saida)
    G, theta = obter_funcao_transferencia(k, tau, theta)

    # Comparação 
    comparar_modelo_saida(Tempo, Saida, G, theta)

    comparar_pade_vs_padding(k, tau, theta)

    comparar_pade_ordens(k, tau, theta)


    # Visualizar dados
    # plot_entrada_saida(Tempo, Entrada, Saida)
    
    # Sintonia CHR
    Kp_chr, Ti_chr, Td_chr = sintonia_pid_chr_sobressinal(k, tau, theta)
    t_chr, y_chr = simular_pid(Tempo, G, theta, Kp_chr, Ti_chr, Td_chr)

    # Sintonia Cohen-Coon
    Kp_cc, Ti_cc, Td_cc = sintonia_pid_cohen_coon(k, tau, theta)
    t_cc, y_cc = simular_pid(Tempo, G, theta, Kp_cc, Ti_cc, Td_cc)

    # Plot comparativo PID
    plt.plot(t_chr, y_chr, label='CHR com sobrevalor')
    plt.plot(t_cc, y_cc, label='Cohen-Coon', linestyle='--')
    plt.plot(Tempo, Saida / np.max(Saida), label='Saída real', alpha=0.5)
    plt.title("Resposta do Sistema com Controladores PID")
    plt.xlabel("Tempo")
    plt.ylabel("Saída")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

    # Avaliação
    print("\n📊 Avaliação CHR:")
    avaliar_desempenho(t_chr, y_chr)

    print("\n📊 Avaliação Cohen-Coon:")
    avaliar_desempenho(t_cc, y_cc)


'''
CÓDIGO INICIAL - TIAGO

from scipy.io import loadmat, whosmat
import numpy as np
import matplotlib.pyplot as plt

# Load do Dataset_Grupo3.mat no diretório atual
# O arquivo deve estar no mesmo diretório que este script ou você deve fornecer o caminho completo para o arquivo4

data = loadmat('Dataset_Grupo3.mat')
data = data.get(whosmat('Dataset_Grupo3.mat')[0][0])[0][0]

Tempo, Entrada, Saida, QuantidadeFisica, Unidades = data
Tempo = Tempo[0]
Entrada = Entrada[0]
Saida = Saida[0]
QuantidadeFisica = QuantidadeFisica[0]
Unidades = Unidades[0]

plt.plot(Tempo, Entrada, label='Entrada')
plt.plot(Tempo, Saida, label='Saída')
plt.show()
print(Tempo)'''















