import scipy.io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Load do Dataset_Grupo3.mat no diretório atual
# O arquivo deve estar no mesmo diretório que este script ou você deve fornecer o caminho completo para o arquivo4

data = scipy.io.loadmat('Dataset_Grupo3.mat')

# A variável 'data' agora contém os dados do arquivo .mat
# Você pode acessar os dados usando a chave correspondente no dicionário 'data'
# Exemplo: se o arquivo contém uma variável chamada 'X', você pode acessá-la assim:
# X = data['X']
# Se você não souber os nomes das variáveis no arquivo, pode usar o seguinte código para listar todas as chaves:
# for key in data.keys():
#     print(key)

print (data)

# Se você quiser salvar os dados em um arquivo CSV, você pode usar o pandas
# Primeiro, instale o pandas se ainda não o tiver instalado:
# pip install pandas





