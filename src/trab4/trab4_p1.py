# Instalar python 3.11 e bibliotecas:
# python3.11 -m venv myenv
# myenv\Scripts\activate
# pip install tensorflow==2.16.1 scikit-learn pandas numpy scikeras

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.initializers import RandomUniform
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import BinaryAccuracy
from tensorflow.keras.metrics import BinaryAccuracy
from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

# Carregar os dados
dados = pd.read_csv('dados_breast.csv')
rotulos = pd.read_csv('rotulos_breast.csv')

# 1) Carregue a base de dados, faça a divisição de treino e teste (para isso, utilize a função train_test_split do sklearn), como o tamanho da base de teste de 0.25.
X_treinamento, X_teste, y_treinamento, y_teste = train_test_split(
    dados, rotulos, test_size=0.25, random_state=42
)

# Pra normalizar os dados:
# scaler = StandardScaler()
# X_treinamento = scaler.fit_transform(X_treinamento)
# X_teste = scaler.transform(X_teste)

def estrutura_da_rede_neural_artificial_e_teste():
    print("--- Estrutura da Rede Neural Artificial e Teste ---\n")

    # 2) Crie a RNA com as seguintes configurações:
    # a) Camada de entrada com 30 neurônios
    # b) Camada oculta densa com 16 neurônios
    # c) Camada oculta com a função de ativação relu e inicialize os pesos com o Random uniform initializer
    # d) Camada de saída com função sigmoid
    rede_neural = Sequential()
    rede_neural.add(Dense(
        units=16, 
        activation='relu', 
        kernel_initializer=RandomUniform(minval=-0.5, maxval=0.5),
        input_dim=30
    ))
    rede_neural.add(Dense(
        units=1, 
        activation='sigmoid'
    ))
    
    # 5) Adicione um otimizador Adam e especifique a classe loss binário - binary crossentropy e a classe metrics para utilizar a métrica de avaliação de acurácia binária
    otimizador = Adam(learning_rate=0.001)
    rede_neural.compile(
        optimizer=otimizador,
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    # 3) Porque utilizamos a classe chamada Sequential para a RNA?
    # A classe Sequential é utilizada porque permite criar modelos camada por camada de forma linear, onde a saída de uma camada é a entrada da próxima, o que é adequado para redes feedforward simples.

    # 6) Para que servem os otimizadores? Como o otimizador Adam funciona?

    # Otimizadores são algoritmos que ajustam os pesos da rede para minimizar a função de perda.
    # O Adam combina as vantagens de dois outros otimizadores (AdaGrad e RMSProp), usando:
    #   - Taxas de aprendizagem adaptativas para cada parâmetro
    #   - Médias móveis dos gradientes (momentum)
    # É eficiente em problemas com muitos dados ou parâmetros.
        
    # 4) A partir da RNA gerada, explique o que são os valores apresentados na tabela da rede_neural.summary()
    # O summary() mostra as camadas da rede, com número de neurônios e número de parâmetros (pesos + bias) por camada e por total
    rede_neural.summary()

    # 7 A) Depois de estruturados os parâmtros da RNA, utilize a função .fit para fazer o treinamento da rede. Como foi utilizado o otimizador Adam, e ele é baseado na descida do gradiente estocástica, é possível definir a quantidade de registros que serão enviados para a RNA, isto é, em cada batch serão utilizados 10 registros.
    rede_neural.fit(
        X_treinamento, y_treinamento,
        batch_size=10,
        # 8) Por fim, defina o número de épocas em que ocorre o treinamento igual a 100.
        epochs=100,
        verbose=0
    )

    # 7 B) Quantos batches serão utilizados ao total?
    # O número total de batches é calculado como (Número de exemplos de treino / batch_size (10)) * número de épocas.

    # 9 A) Crie uma variável chamada previsoes para realizar a previsão dos dados de teste (X_teste) O resultado da rede deve ser um valor entre 0 e 1.
    previsoes = rede_neural.predict(X_teste)
    print("Previsões: ", previsoes[:10])
    # 9 B) Porque isso acontece?
    # A saída está entre 0 e 1 porque usamos a função sigmoid na camada de saída, que converte a saída para este intervalo.

    # 10) Conversão para valores binários
    previsoes_binarias = (previsoes > 0.5).astype(int)
    print("Previsões convertidas: ", previsoes_binarias[:10])

    # Avaliação final
    previsoes = rede_neural.predict(X_teste)
    print(f"Previsões extremas: {np.sum(previsoes < 0.01)} (<0.01), {np.sum(previsoes > 0.99)} (>0.99) | Total: {len(previsoes)}")

    loss, accuracy = rede_neural.evaluate(X_teste, y_teste)
    print(f"Perda no teste: {loss*100:.2f}%")
    print(f"Acurácia no teste: {accuracy*100:.2f}%")

    # 9) Resposta: O resultado mostra a performance da RNA nos dados de teste:
    # Loss (perda): É o valor da função de custo ao aplicar o modelo nos dados de teste. Mede o quanto o modelo errou nas previsões.
    # Accuracy (acurácia): É a proporção de acertos do modelo nos dados de teste. Varia entre 0 e 1. Mede quantas vezes a rede acertou a classe prevista.

def camadas_e_otimizacao_da_rna():
    print("--- Camadas e Otimização da RNA --- \n")

    rede_neural = Sequential()
    
    rede_neural.add(Dense(
        units=16, 
        activation='relu', 
        kernel_initializer=RandomUniform(minval=-0.5, maxval=0.5),
        input_dim=30
    ))
    
    ## 10 A) Adicione mais uma camada para oculta densa com 16 neurônio com a função de ativação relu e inicialize os pesos de utilize o Random uniform initializer.
    rede_neural.add(Dense(
        units=16, 
        activation='relu', 
        kernel_initializer=RandomUniform(minval=-0.5, maxval=0.5)
    ))
    
    rede_neural.add(Dense(
        units=1, 
        activation='sigmoid'
    ))
    
    """
    11) Abaixo são adicionados os parâmetros do otimizador, que são a taxa de aprendizado e o clipvalue. O que eles fazem?

    - learning_rate (taxa de aprendizado): Controla o tamanho do passo que o otimizador dá a cada atualização dos pesos. 
    Um valor muito alto pode fazer o modelo ter resultados piores, enquanto um valor muito baixo pode tornar o treinamento muito lento.
    
    - clipvalue: Limita o valor absoluto dos gradientes durante o treinamento. Isso ajuda a prevenir gradientes muito fora da curva, 
    que podem atrapalhar o treinamento. Neste caso, os gradientes ficam no intervalo de 0.5 até -0.5.
    """
    otimizador = Adam(learning_rate=0.001, clipvalue=0.5)
    rede_neural.compile(
        optimizer=otimizador,
        loss='binary_crossentropy',
        metrics=['binary_accuracy']
    )
        
    rede_neural.summary()

    """
    10 B) Qual o total de parâmetros da RNA agora?

    Com a adição da segunda camada oculta, o número total de parâmetros aumentou:
    - Primeira camada oculta: (30 inputs * 16 neurônios) + 16 bias = 496 parâmetros
    - Segunda camada oculta : (16 inputs * 16 neurônios) + 16 bias = 272 parâmetros
    - Camada de saída       : (16 inputs *  1 neurônio ) +  1 bias =  17 parâmetros
    Total: 785 parâmetros
    """

    ## 12 A) Teste novamente a RNA
    rede_neural.fit(
        X_treinamento, y_treinamento,
        batch_size=10,
        epochs=100,
        verbose=0
    )

    previsoes = rede_neural.predict(X_teste)
    print(f"Previsões extremas: {np.sum(previsoes < 0.01)} (<0.01), {np.sum(previsoes > 0.99)} (>0.99) | Total: {len(previsoes)}")

    loss, accuracy = rede_neural.evaluate(X_teste, y_teste)
    print(f"Perda no teste: {loss*100:.2f}%")
    print(f"Acurácia no teste: {accuracy*100:.2f}%")

    # Pra testar se aconteceu overfitting:
    # train_loss, train_acc = rede_neural.evaluate(X_treinamento, y_treinamento, verbose=0)
    # test_loss, test_acc = rede_neural.evaluate(X_teste, y_teste, verbose=0)

    # print(f"Treino - Loss: {train_loss:.4f} | Acurácia: {train_acc*100:.2f}%")
    # print(f"Teste  - Loss: {test_loss:.4f}  | Acurácia: {test_acc*100:.2f}%")

    # Pra testar se binary_crossentropy penalizou previsões erradas com confiança alta:
    # y_teste_array = np.array(y_teste).reshape(-1, 1)

    # limiar_inferior = 0.01
    # limiar_superior = 0.99

    # confiantes_0 = previsoes < limiar_inferior
    # confiantes_1 = previsoes > limiar_superior

    # erradas_confiantes_0 = confiantes_0 & (y_teste_array == 1)  # Previu ≈0 mas era 1
    # erradas_confiantes_1 = confiantes_1 & (y_teste_array == 0)  # Previu ≈1 mas era 0

    # print(f"Previu 0 mas era 1 (erro): {np.sum(erradas_confiantes_0)}/{np.sum(confiantes_0)}")
    # print(f"Previu 1 mas era 0 (erro): {np.sum(erradas_confiantes_1)}/{np.sum(confiantes_1)}")

    """
    Resposta 12 B) Aumentar a quantidade de camadas melhorou ou piorou os resultados? Explique o que aconteceu com a RNA e porque.

    Adicionar a segunda camada oculta aumentou muito o loss (de ~133.05%% pra ~761.94%) e diminuiu um pouco a acurácia (de ~88% pra ~82%).
    
    O loss aumentou provavelmente pq a função binary_crossentropy penaliza muito previsões erradas mas confiantes (ex.: prever 0.99 quando o correto é 0).
    """

estrutura_da_rede_neural_artificial_e_teste()
camadas_e_otimizacao_da_rna()

# ## 13-14) K-Fold Cross Validation
# # Carregando os dados completos novamente para o K-Fold
# X = dados.values
# y = rotulos.values.ravel()  # Convertendo para array 1D

# def criar_rede():
#     k.clear_session()  # Limpa a sessão do TensorFlow/Keras
#     rede_neural = Sequential([
#         Dense(units=16, activation='relu', kernel_initializer='random_uniform', input_shape=(30,)),
#         Dense(units=16, activation='relu', kernel_initializer='random_uniform'),
#         Dense(units=1, activation='sigmoid')
#     ])
#     otimizador = Adam(learning_rate=0.001, clipvalue=0.5)
#     rede_neural.compile(
#         optimizer=otimizador,
#         loss='binary_crossentropy',
#         metrics=['binary_accuracy']
#     )
#     return rede_neural

# # Configurando o KerasClassifier para usar com cross_val_score
# rede_neural = KerasClassifier(
#     model=criar_rede,
#     epochs=100,
#     batch_size=10,
#     verbose=0
# )

# # Executando K-Fold Cross Validation com 10 folds
# resultados = cross_val_score(
#     estimator=rede_neural,
#     X=X,
#     y=y,
#     cv=10,  # 10 folds
#     scoring='accuracy'
# )

# # Resultados
# print("\nAcurácias em cada fold:", resultados)
# print("Média das acurácias:", resultados.mean())
# print("Desvio padrão das acurácias:", resultados.std())

# """
# Resposta 13:
# O K-Fold Cross Validation foi configurado da seguinte forma:
# 1. A função criar_rede define a arquitetura da RNA (igual à usada anteriormente)
# 2. Usamos KerasClassifier para adaptar o modelo Keras à interface do scikit-learn
# 3. cross_val_score divide os dados em 10 folds (cv=10), treina em 9 e testa em 1, repetindo para todos os folds
# 4. A métrica usada é acurácia ('accuracy')

# Isso fornece uma estimativa mais robusta do desempenho do modelo, pois usa todas as amostras para treino e teste em diferentes combinações.

# Resposta 14:
# O desvio padrão dos resultados mostra a variabilidade no desempenho entre os diferentes folds:
# - Um desvio padrão baixo indica que o modelo tem desempenho consistente em diferentes subconjuntos dos dados
# - Um desvio padrão alto sugere que o modelo pode ser sensível à seleção específica de dados de treino/teste
# Combinado com a média de acurácia, podemos ter mais confiança na generalização do modelo se ambos os valores forem bons.
# """