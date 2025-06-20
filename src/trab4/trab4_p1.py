# Instalar python 3.11 e bibliotecas:
# python3.11 -m venv myenv
# myenv\Scripts\activate
# pip install tensorflow==2.16.1 scikit-learn pandas numpy scikeras

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.initializers import RandomUniform
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import BinaryAccuracy
from tensorflow.keras.metrics import BinaryAccuracy
from scikeras.wrappers import KerasClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from tensorflow.keras import backend as k

# Carregar os dados
dados = pd.read_csv('dados_breast.csv')
rotulos = pd.read_csv('rotulos_breast.csv')

# 1) Carregue a base de dados, faça a divisição de treino e teste (para isso, utilize a função train_test_split do sklearn), como o tamanho da base de teste de 0.25.
X_treinamento, X_teste, y_treinamento, y_teste = train_test_split(
    dados, rotulos, test_size=0.25, random_state=42
)

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
  
    k.clear_session()

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

def k_fold_cross_validation():
    k.clear_session()

    rede_neural = Sequential([
        tf.keras.layers.InputLayer(shape=(30,)),
        tf.keras.layers.Dense(units=16, activation='relu', kernel_initializer='random_uniform'),
        tf.keras.layers.Dense(units=16, activation='relu', kernel_initializer='random_uniform'),
        tf.keras.layers.Dense(units=1, activation = 'sigmoid')])

    otimizador = tf.keras.optimizers.Adam(learning_rate = 0.001, clipvalue = 0.5)

    rede_neural.compile(optimizer = otimizador, loss = 'binary_crossentropy', metrics = ['binary_accuracy'])

    # 13) Explique como esta rede foi configurada e como é possível chegar no resultado dela. O que é necessário fazer?
    # A rede foi configurada do mesmo jeito que antes, mas agora a gente usa o KerasClassifier pra poder chamar a função cross_val_score (do sklearn) no objeto da classe Sequential (do tensorflow).
    # O cross_val_score:
    #   divide os dados em 10 partes (folds)
    #   faz 10 rodadas:
    #       Treina a rede com 9 partes.
    #       Testa com a 10ª parte.
    #   faz a média das precisões das rodadas
    
    rede_neural = KerasClassifier(model = rede_neural, epochs = 100, batch_size = 10)

    X = dados.values
    y = rotulos.values.ravel()
    resultados = cross_val_score(estimator = rede_neural, X = X, y = y, cv = 10, scoring = 'accuracy')

    print("\nAcurácias em cada fold:", resultados)
    print("Média das acurácias:", resultados.mean())
    print("Desvio padrão das acurácias:", resultados.std())

    # 14) Calcule também o Desvio Padrão dos resultados para avaliar o modelo. O que é possível concluir com esse resultado?
    # O desvio padrão mede o quanto as acurácias dos folds variaram entre si. Um desvio padrão de 7,6% não é muito baixo, então o desempenho da rede variou bastante entre os folds.
    # Isso pode ser um sinal de overfitting.

def overfitting_e_dropout():
    k.clear_session()

    rede_neural = Sequential([
        tf.keras.layers.InputLayer(shape=(30,)),
        tf.keras.layers.Dense(units=16, activation='relu', kernel_initializer='random_uniform'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(units=16, activation='relu', kernel_initializer='random_uniform'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(units=1, activation = 'sigmoid')])

    otimizador = tf.keras.optimizers.Adam(learning_rate = 0.001, clipvalue = 0.5)

    rede_neural.compile(optimizer = otimizador, loss = 'binary_crossentropy', metrics = ['binary_accuracy'])

    rede_neural = KerasClassifier(model = rede_neural, epochs = 100, batch_size = 10)

    X = dados.values
    y = rotulos.values.ravel()
    resultados = cross_val_score(estimator = rede_neural, X = X, y = y, cv = 10, scoring = 'accuracy')

    # 14) Aplique o dropout de 20% na primeira e segunda camada oculta. O que acontece com os resultados? E o Desvio Padrão?
    # A acurácia dos folds aumentou um pouco, e o desvio padrão entre elas diminuiu.
    print("\nAcurácias em cada fold:", resultados)
    print("Média das acurácias:", resultados.mean())
    print("Desvio padrão das acurácias:", resultados.std())

def tuning_dos_hiperparametros():
    def criar_rede(optimizer, loss, kernel_initializer, activation, neurons):
        k.clear_session()
        rede_neural = Sequential([
            tf.keras.layers.InputLayer(shape=(30,)),
            tf.keras.layers.Dense(units=neurons, activation=activation, kernel_initializer=kernel_initializer),
            tf.keras.layers.Dropout(rate = 0.2),
            tf.keras.layers.Dense(units=neurons, activation=activation, kernel_initializer=kernel_initializer),
            tf.keras.layers.Dropout(rate = 0.2),
            tf.keras.layers.Dense(units=1, activation = 'sigmoid')])
        rede_neural.compile(optimizer = optimizer, loss = loss, metrics = ['binary_accuracy'])
        return rede_neural

    rede_neural = KerasClassifier(model = criar_rede)

    # 15) Descreva como a RNA foi configurada para fazer o processo de tuning.
    parametros = {
        'batch_size': [10, 30],
        'epochs': [50],
        'model__optimizer': ['adam'],
        'model__loss': ['binary_crossentropy'],
        'model__kernel_initializer': ['random_uniform', 'normal'],
        'model__activation': ['relu'],
        'model__neurons': [16]
    }

    grid_normal = GridSearchCV(estimator = rede_neural, param_grid = parametros, scoring = 'accuracy', cv = 5)

    X_raw = dados.values
    y = rotulos.values.ravel()
    grid_normal.fit(X_raw, y)

    # 2. Grid search com os dados normalizados (Z-score)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    grid_normalizada = GridSearchCV(estimator=rede_neural, param_grid=parametros, scoring='accuracy', cv=5)
    grid_normalizada.fit(X_scaled, y)

    # Resultados
    print("\nSem normalização:")
    print("  Melhor acurácia:", grid_normal.best_score_)
    print("  Parâmetros:", grid_normal.best_params_)

    print("\nCom normalização (Z-score):")
    print("  Melhor acurácia:", grid_normalizada.best_score_)
    print("  Parâmetros:", grid_normalizada.best_params_)

    grid_normal.best_estimator_.model_.save("modelo_treiando_sem_normalizacao.keras")
    grid_normal.best_estimator_.model_.save("modelo_treiando_com_normalizacao.keras")

def carrega_e_mostra_modelos_salvos():
    print("\n--- Carregando Modelos Salvos---\n")
    
    try:
        modelo_sem_normalizacao = tf.keras.models.load_model("modelo_treiando_sem_normalizacao.keras")
        modelo_sem_normalizacao.summary()
    except Exception as e:
        print(f"Erro ao carregar modelo_treiando_sem_normalizacao.keras: {e}")
    
    print("\n" + "-"*60 + "\n")
    
    try:
        modelo_com_normalizacao = tf.keras.models.load_model("modelo_treiando_com_normalizacao.keras")
        modelo_com_normalizacao.summary()
    except Exception as e:
        print(f"Erro ao carregar modelo_treiando_com_normalizacao.keras: {e}")


# estrutura_da_rede_neural_artificial_e_teste()
# camadas_e_otimizacao_da_rna()
# k_fold_cross_validation()
# overfitting_e_dropout()
# tuning_dos_hiperparametros()
carrega_e_mostra_modelos_salvos()