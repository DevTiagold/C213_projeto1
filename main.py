from scipy.io import loadmat, whosmat
import numpy as np
import matplotlib.pyplot as plt
#from matplotlib.widgets import Slider, Button -->  usar quando for implementar interface
import control as ctrl



def load_dados(filename):
    data = loadmat(filename)
    data = data.get(whosmat(filename)[0][0])[0][0]
    Tempo, Entrada, Saida, QuantidadeFisica, Unidades = data
    #pegando o valor medio da entrada
    return Tempo[0], Entrada[0], Saida[0], QuantidadeFisica[0], Unidades[0]


def plot_entrada_saida(tempo, entrada, saida):   
    plt.figure(figsize=(10, 5))
    plt.plot(tempo, entrada, label='Entrada')
    plt.plot(tempo, saida, label='Saída', linestyle='--')
    plt.title('Curva de Entrada e Saída do Sistema')
    plt.xlabel('Tempo [s]')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.legend()
    plt.xlim(0, 30000) 
    plt.tight_layout()
    plt.show()

def identificar_planta_smith(tempo, entrada, saida):
    
    #entrada_inicial = np.min(entrada)
   # entrada_final = np.max(entrada)
    saida_inicial = saida[0]   # Primeiro valor
    saida_final = saida[-1]       # Último valor
    print(saida_final)
    delta_entrada = entrada.mean()
    #print(f"delta_entrada: {delta_entrada}")
    delta_saida = saida_final - saida_inicial
    print(f"delta_saida: {delta_saida}")    
    if delta_saida == 0 or delta_entrada == 0:
        raise ValueError("A variação de entrada ou saída é zero. Verifique se o sistema respondeu ao degrau.")

    # Ganho estático
    k = (saida_final - saida_inicial) / delta_entrada
    print(f"Ganho (k): {k:.4f}")
    # Valores correspondentes a 28.3% e 63.2% da resposta máxima
    y_283 = saida_inicial + 0.283 * delta_saida
    y_632 = saida_inicial + 0.632 * delta_saida

    # Verificar se y_283 e y_632 estão dentro do intervalo da saída
    if not (saida_inicial <= y_283 <= saida_final) or not (saida_inicial <= y_632 <= saida_final):
        raise ValueError("Os níveis de 28.3% ou 63.2% estão fora do intervalo da saída. A resposta pode não ter chegado ao patamar.")

    # Interpola os tempos correspondentes aos pontos de 28.3% e 63.2%
    t1 = np.interp(y_283, saida, tempo)
    t2 = np.interp(y_632, saida, tempo)

    # Cálculo da constante de tempo e atraso (Smith)
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
    
    k = delta_saida / entrada.mean()
    
    #print(f"Ganho (k): {k:.4f}")

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
    t_padding, y_padding = ctrl.step_response(G_padding * amp_degrau, T)
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

    # Padé ordem 20 --> ideal
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


# Comparação dos modelos de saída
def comparar_modelo_saida(tempo, saida_real, G, theta):
    t_sim, y_sim = ctrl.step_response(G, T=tempo)

    atraso_amostras = np.searchsorted(tempo, tempo[0] + theta)
    y_sim_atrasado = np.concatenate((np.zeros(atraso_amostras), y_sim))[:len(tempo)]

    # Patamar inicial e final
    saida_inicial = np.min(saida_real)
    saida_final = np.max(saida_real)
    delta_saida = saida_final - saida_inicial

    # Cálculo dos pontos de 28.3% e 63.2% da resposta
    y_283 = saida_inicial + (0.283 * delta_saida)
    y_632 = saida_inicial + (0.632 * delta_saida)

    # Interpola os tempos correspondentes a esses valores
    t1 = np.interp(y_283, saida_real, tempo)
    t2 = np.interp(y_632, saida_real, tempo)

    # Gráfico
    plt.figure(figsize=(10, 5))
    plt.plot(tempo, saida_real, label='Saída Real', color='tab:blue')
    plt.plot(tempo, y_sim_atrasado, '--', label='Modelo 1ª ordem (Smith)', color='tab:orange')

    # Destaques visuais
    plt.axhline(y_283, color='green', linestyle=':', label='28.3% da resposta')
    plt.axhline(y_632, color='purple', linestyle=':', label='63.2% da resposta')
    plt.axvline(t1, color='green', linestyle='--')
    plt.axvline(t2, color='purple', linestyle='--')

    plt.text(t1, y_283 + 10, f"t1 ≈ {t1:.0f}s", color='green')
    plt.text(t2, y_632 + 10, f"t2 ≈ {t2:.0f}s", color='purple')

    plt.title('Comparação da resposta: Real vs Modelo de Smith')
    plt.xlabel('Tempo [s]')
    plt.ylabel('Saída')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Erro médio
    erro = np.mean(np.abs(saida_real - y_sim_atrasado))
    print(f"\nErro médio entre o modelo e a saída real: {erro:.4f}")

# Sintonizar PID 
# CHR com Sobrevalor e Cohen-Coon

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


# Execução da main
if __name__ == "__main__":

    # ETAPA 1 – Carregar e visualizar os dados
    Tempo, Entrada, Saida, QuantidadeFisica, Unidades = load_dados('Dataset_Grupo3.mat')
    amp_degrau = Entrada.mean()
    Tempo = Tempo.astype(float)
    plot_entrada_saida(Tempo, Entrada, Saida)

    # ETAPA 2 – Identificação da planta
    k, tau, theta = identificar_planta_smith(Tempo, Entrada, Saida)
    G, theta = obter_funcao_transferencia(k, tau, theta)

    k_sund, tau_sund, theta_sund = identificar_planta_sundaresan(Tempo, Entrada, Saida)
    G_sund = obter_funcao_transferencia_pade(k_sund, tau_sund, theta_sund, ordem=1)

    # ETAPA 3 – Simulação das respostas dos modelos identificados
    t_sund, y_sund = ctrl.step_response(G_sund*amp_degrau, T=Tempo)

    t_smith, y_smith = ctrl.step_response(G*amp_degrau, T=Tempo)
    atraso_smith = np.searchsorted(Tempo, Tempo[0] + theta)
    y_smith_atrasado = np.concatenate((np.zeros(atraso_smith), y_smith))[:len(Tempo)]

    # ETAPA 4 – Comparações visuais dos mod3los
    # Real vs Sundaresan
    plt.figure(figsize=(10, 5))
    plt.plot(Tempo, Saida, label='Saída Real', color='tab:blue')
    plt.plot(t_sund, y_sund, '--', label='Modelo 1ª ordem - Sundaresan', color='tab:orange')
    plt.title('Comparação da resposta: Real vs Modelo de Sundaresan')
    plt.xlabel('Tempo [s]')
    plt.ylabel('Saída')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Smith vs Sundaresan
    plt.figure(figsize=(10, 5))
    plt.plot(Tempo, Saida, label='Saída Real', color='tab:blue')
    plt.plot(Tempo, y_smith_atrasado, '--', label='Modelo Smith - Padé', color='tab:green')
    plt.plot(t_sund, y_sund, '--', label='Modelo Sundaresan - Padé', color='tab:red')
    plt.title("Comparação dos modelos: Smith vs Sundaresan")
    plt.xlabel("Tempo [s]")
    plt.ylabel("Saída")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # ETAPA 5 – Comparações adicionais
    comparar_modelo_saida(Tempo, Saida, G, theta)
    comparar_pade_vs_padding(k, tau, theta)
    comparar_pade_ordens(k, tau, theta)

    # ETAPA 6 – Sintonias PID
    Kp_chr, Ti_chr, Td_chr = sintonia_pid_chr_sobressinal(k, tau, theta)
    t_chr, y_chr = simular_pid(Tempo, G, theta, Kp_chr, Ti_chr, Td_chr)

    Kp_cc, Ti_cc, Td_cc = sintonia_pid_cohen_coon(k, tau, theta)
    t_cc, y_cc = simular_pid(Tempo, G, theta, Kp_cc, Ti_cc, Td_cc)

    # ETAPA 7 – Resposta com controladores PID (??)
    plt.figure()
    plt.plot(t_chr, y_chr, label='CHR com sobrevalor')
    plt.plot(t_cc, y_cc, label='Cohen-Coon', linestyle='--')
    plt.plot(Tempo, Saida, label='Saída real', alpha=0.5)
    plt.title("Resposta do Sistema com Controladores PID")
    plt.xlabel("Tempo")
    plt.ylabel("Saída")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

    # ETAPA 8 – Avaliação de overshoot e tempo de acomodação
    print("\n Avaliação CHR:")
    avaliar_desempenho(t_chr, y_chr)

    print("\n Avaliação Cohen-Coon:")
    avaliar_desempenho(t_cc, y_cc)
"""
def identificar_planta(tempo, entrada, saida, metodo='smith'):

    Identifica os parâmetros da planta (ganho, constante de tempo, atraso) usando
    o método especificado: 'smith' ou 'sundaresan'.

    Parâmetros:
        tempo   : array de tempo
        entrada : sinal de entrada
        saida   : sinal de saída
        metodo  : 'smith' ou 'sundaresan'

    Retorna:
        k       : ganho estático
        tau     : constante de tempo
        theta   : atraso (tempo morto)
    

    entrada_inicial = np.mean(entrada[:10])
    entrada_final = np.mean(entrada[-10:])
    saida_inicial = np.mean(saida[:10])
    saida_final = np.mean(saida[-10:])
    
    delta_entrada = entrada.mean() if metodo == 'smith' else (entrada_final - entrada_inicial)
    delta_saida = saida_final - saida_inicial

    if delta_saida == 0 or delta_entrada == 0:
        raise ValueError("A variação de entrada ou saída é zero. Verifique se o sistema respondeu ao degrau.")

    k = delta_saida / delta_entrada

    if metodo.lower() == 'smith':
        y_283 = saida_inicial + 0.283 * delta_saida
        y_632 = saida_inicial + 0.632 * delta_saida

        if not (saida_inicial <= y_283 <= saida_final) or not (saida_inicial <= y_632 <= saida_final):
            raise ValueError("Níveis de 28.3% ou 63.2% fora do intervalo da saída. A resposta pode não ter estabilizado.")

        t1 = np.interp(y_283, saida, tempo)
        t2 = np.interp(y_632, saida, tempo)

        tau = 1.5 * (t2 - t1)
        theta = t1 - 0.3 * tau

        print(f"\nIdentificação via Método de Smith:")
    
    elif metodo.lower() == 'sundaresan':
        saida_norm = (saida - saida_inicial) / delta_saida
        y1, y2 = 0.353, 0.853

        t1 = tempo[np.where(saida_norm >= y1)[0][0]]
        t2 = tempo[np.where(saida_norm >= y2)[0][0]]

        tau = 0.67 * (t2 - t1)
        theta = 1.3 * t1 - 0.29 * t2

        print(f"\nIdentificação via Método de Sundaresan:")
    
    else:
        raise ValueError("Método inválido. Escolha 'smith' ou 'sundaresan'.")

    print(f"Ganho (k): {k:.4f}")
    print(f"Constante de tempo (tau): {tau:.2f} s")
    print(f"Atraso (theta): {theta:.2f} s")

    return k, tau, theta
"""
