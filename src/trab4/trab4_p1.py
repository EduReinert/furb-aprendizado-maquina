# Instalar python 3.11 e bibliotecas:
# python3.11 -m venv myenv
# myenv\Scripts\activate
# pip install tensorflow==2.16.1 scikit-learn pandas numpy scikeras

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.initializers import RandomUniform
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import BinaryAccuracy
from tensorflow.keras.metrics import BinaryAccuracy
from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import cross_val_score

# Carregar os dados
dados = pd.read_csv('dados_breast.csv')
rotulos = pd.read_csv('rotulos_breast.csv')

# 1) Carregue a base de dados, faça a divisição de treino e teste (para isso, utilize a função train_test_split do sklearn), como o tamanho da base de teste de 0.25.
X_treinamento, X_teste, y_treinamento, y_teste = train_test_split(
    dados, rotulos, test_size=0.25, random_state=42
)

# Normalização dos dados
# scaler = StandardScaler()
# X_treinamento = scaler.fit_transform(X_treinamento)
# X_teste = scaler.transform(X_teste)

# 2) Criação da RNA
def criar_rna():
    rede_neural = Sequential()
    
    # a) Camada de entrada com 30 neurônios (uma para cada feature)
    # b) Camada oculta densa com 16 neurônios
    # c) Função de ativação relu e inicialização dos pesos
    rede_neural.add(Dense(
        units=16, 
        activation='relu', 
        kernel_initializer=RandomUniform(minval=-0.5, maxval=0.5),
        input_dim=30
    ))
    
    # d) Camada de saída com função sigmoid
    rede_neural.add(Dense(
        units=1, 
        activation='sigmoid'
    ))
    
    # 5) Otimizador e configurações
    # 6) Resposta: Otimizadores são algoritmos que ajustam os pesos da rede para minimizar a função de perda.
    #   O Adam combina as vantagens de dois outros otimizadores (AdaGrad e RMSProp), usando:
    #   - Taxas de aprendizagem adaptativas para cada parâmetro
    #   - Médias móveis dos gradientes (momentum)
    #   É eficiente em problemas com muitos dados ou parâmetros.
    otimizador = Adam(learning_rate=0.001)
    rede_neural.compile(
        optimizer=otimizador,
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return rede_neural

rede_neural = criar_rna()

# 3) Resposta: A classe Sequential é utilizada porque permite criar modelos camada por camada de forma linear, 
# onde a saída de uma camada é a entrada da próxima, o que é adequado para redes feedforward simples.

# 4) Resposta: O summary() mostra:
# - as camadas da rede, com número de neurônios e número de parâmetros (pesos + bias)

rede_neural.summary()

print(X_treinamento.shape)
# 7) Treinamento da rede
historico = rede_neural.fit(
    X_treinamento, y_treinamento,
    batch_size=10,  # 10 registros por batch
    epochs=100,
    verbose=1
)


# 7) Resposta: O número total de batches é calculado como:
# (Número de exemplos de treino / batch_size (10)) * número de épocas.

# 9) Previsões
previsoes = rede_neural.predict(X_teste)

# 9) Resposta: A saída está entre 0 e 1 porque usamos a função sigmoid na camada de saída,
# que comprime a saída para este intervalo, interpretável como probabilidade.

# 10) Conversão para valores binários
previsoes_binarias = (previsoes > 0.5).astype(int)

# Avaliação final
loss, accuracy = rede_neural.evaluate(X_teste, y_teste)
print(f"Perda no teste: {loss*100:.2f}%")
print(f"Acurácia no teste: {accuracy*100:.2f}%")

# 9) Resposta: O resultado mostra a performance da RNA nos dados de teste:
# - Loss (perda): valor da função de custo (quanto menor melhor)
# - Accuracy (acurácia): porcentagem de previsões corretas