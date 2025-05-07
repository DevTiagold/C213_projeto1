
# 🔧 Identificação de Sistemas e Sintonização PID com Aproximação de Padé

Este projeto realiza a **identificação de um sistema de 1ª ordem com atraso** a partir de dados reais de entrada e saída, utilizando os métodos de **Smith** e **Sundaresan**. A modelagem inclui a **aproximação de Padé** para representar o atraso puro, além de implementar **controladores PID** com sintonias CHR e Cohen-Coon.

## 📂 Estrutura do Projeto

- `Dataset_Grupo3.mat`: Arquivo com os dados de tempo, entrada (degrau) e saída (resposta do sistema).
- `main.py`: Script principal com as seguintes etapas:
  - Leitura e visualização dos dados.
  - Identificação da planta (Smith e Sundaresan).
  - Modelagem da função de transferência com Padé.
  - Simulação da resposta ao degrau.
  - Sintonização e simulação de controladores PID.
  - Avaliação de desempenho (overshoot e tempo de acomodação).
- Funções auxiliares para:
  - Aproximação de Padé.
  - Comparação de respostas.
  - Cálculo de erro médio.

## ▶️ Como Executar

1. **Clone o repositório**:
   ```bash
   git clone https://github.com/seu-usuario/seu-repositorio.git
   cd seu-repositorio


## Instale as dependências:


pip install numpy matplotlib scipy control

## Execute o script principal:
python main.py

### O script irá:

- Mostrar os gráficos da entrada e saída.

- Exibir os parâmetros identificados (ganho, τ, θ).

- Comparar a resposta dos modelos com a resposta real.

- Plotar a resposta dos controladores PID.

- Calcular o overshoot e tempo de acomodação.

## ⚙️ Técnicas Utilizadas
Identificação da Planta:

- Método de Smith
- Método de Sundaresan

## Modelagem:

- Aproximação de Padé (ordem 20) para simular atraso

- Funções de transferência com control.TransferFunction

## Controle PID:

Sintonias:

- CHR com sobrevalor

- Cohen-Coon

Simulação com control.feedback e control.step_response


## 👨‍💻 Desenvolvido por

Tiago Augusto 
Wiliane Carolina 