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
print(Tempo)















