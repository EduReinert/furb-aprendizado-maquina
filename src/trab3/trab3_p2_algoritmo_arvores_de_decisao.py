import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import pickle
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report
from yellowbrick.classifier import ConfusionMatrix
from sklearn.ensemble import RandomForestClassifier

"""
ALUNOS:

Augusto Arraga
Eduardo Reinert
Vinicius Vanelli
"""
    
def algoritmo_arvore_decisao_menor():
    """# 1 - Importação dos dados Pré-Processados
    a) importe o arquivo salvo como 'risco_credito.pkl'
    """
    with open('risco_credito.pkl', 'rb') as f:
        X_risco_credito, y_risco_credito = pickle.load(f)
    
    """
    # 2 - Algoritmo de Árvore de Decisão
    
    b) Calcule a árvore de decisão, utilizando como critério a entropia.
    Coloque como nome da variável: arvore_risco_credito
    """
        
    arvore_risco_credito = DecisionTreeClassifier()
    arvore_risco_credito = arvore_risco_credito.fit(X_risco_credito, y_risco_credito)

    #c) Utilize o feature_importances_ para retornar a importância de cada atributo. Qual possui o maior ganho de informação?
    print(arvore_risco_credito.feature_importances_)

    """
    d) Gere uma visualização da sua árvore de decisão utilizando o pacote tree da biblioteca do sklearn.
    OBS: Adicione cores, nomes para os atributos e para as classes.
    """
    plt.figure(figsize=(10, 6))
    plot_tree(arvore_risco_credito, 
              feature_names=X_risco_credito,  # Se X_risco_credito for um DataFrame
              class_names=arvore_risco_credito.classes_,  # Nomes das classes
              filled=True,  # Preenchimento com cores
              rounded=True)  # Bordas arredondadas
    plt.show()
    
    
    #e) FAZER A PREVISÃO || Utilize .predict para fazer a previsão realizada no exemplo em sala.
    
    # i. história boa (1), dívida alta (0), garantia nenhuma (0), renda > 35 (2)
    novo_cliente1 = [[1, 0, 0, 2]]
    
    # ii. história ruim (0), dívida alta (0), garantia adequada (1), renda < 15 (0)
    novo_cliente2 = [[0, 0, 1, 0]]
    
    # Fazendo as previsões
    previsao1 = arvore_risco_credito.predict(novo_cliente1)
    previsao2 = arvore_risco_credito.predict(novo_cliente2)
    
    print("\nPrevisões:")
    print(f"i.  história boa, dívida alta, garantia nenhuma, renda > 35: {previsao1[0]}")
    print(f"ii. história ruim, dívida alta, garantia adequada, renda < 15: {previsao2[0]}")

def algoritmo_arvore_decisao_maior():
    """
    #3 - Algoritmo de Árvore de Decisão para uma base de dados maior (Credit Data)

    Nesta seção você deverá testar o uso da Árvore de Decisão para a Base de Dados Credit Risk Dataset. Aqui estaremos analisando os clientes que pagam (classe 0) ou não pagam a dívida (classe 1), a fim do banco conceder empréstimo.

    a) Ao abrir o arquivo utilize .shape para verificar o tamanho dos dados de treinamento e de teste
    """
    # abrir o arquivo
    with open('credit.pkl', 'rb') as f:
        X_credit_treinamento, y_credit_treinamento, X_credit_teste, y_credit_teste = pickle.load(f)

    print("\nDimensões dos dados:")
    print(f"X treinamento: {X_credit_treinamento.shape}")
    print(f"y treinamento: {y_credit_treinamento.shape}")
    print(f"X teste: {X_credit_teste.shape}")
    print(f"y teste: {y_credit_teste.shape}")
    
    # b) Importar e treinar o DecisionTreeClassifier
    arvore_credit = DecisionTreeClassifier(random_state=0)
    arvore_credit.fit(X_credit_treinamento, y_credit_treinamento)
    
    # c) Fazer previsões com os dados de teste
    previsoes = arvore_credit.predict(X_credit_teste)
    
    print("\nPrimeiras 10 previsões vs valores reais:")
    for i in range(10):
        print(f"Previsão: {previsoes[i]} | Real: {y_credit_teste[i]}")
    
    # d) Calcular a acurácia
    acuracia = accuracy_score(y_credit_teste, previsoes)
    print(f"\nAcurácia: {acuracia:.2%}")
    
    """
    #e) Análise da matriz de confusão
    
    i. Quantos clientes foram classificados corretamente que pagam a dívida?

    ii. Quantos clientes foram classificados incorretamente como não pagantes?

    iii. Quantos clientes foram classificados corretamente que não pagam?

    iv. Quantos clientes foram classificados incorretamente como pagantes?
    """
    cm = ConfusionMatrix(arvore_credit)
    cm.fit(X_credit_treinamento, y_credit_treinamento)
    cm.score(X_credit_teste, y_credit_teste)
    plt.title("Matriz de Confusão - Yellowbrick")
    plt.show()
    
    # f) Faça um print do parâmetro classification_report entre os dados de teste e as previsões. Explique qual é a relação entre precision e recall nos dados. Como você interpreta esses dados?
    print("\nClassification Report:")
    report = classification_report(y_credit_teste, previsoes)
    print(report)
    
    
    """ g) Gere uma visualização da sua árvore de decisão utilizando o pacote tree da biblioteca do sklearn.
    OBS 1: Os atributos previsores são = ['income', 'age', 'loan']

    OBS 2: Adicione cores, nomes para os atributos e para as classes. Você pode utilizar a função fig.savefig para salvar a árvore em uma imagem .png

    """
    plt.figure(figsize=(25, 15))
    atributos = ['income', 'age', 'loan']  # Nomes dos atributos previsores
    classes = ['pagador', 'não pagador']    # Nomes das classes
    
    # Plotando a árvore
    plot_tree(arvore_credit,
              feature_names=atributos,
              class_names=classes,
              filled=True,          # Preenchimento com cores
              rounded=True,         # Bordas arredondadas
              proportion=True,      # Mostra proporções nas classes
              impurity=False,       # Não mostra impureza
              fontsize=10,          # Tamanho da fonte
              max_depth=3)          # Limitando a profundidade para melhor visualização
    
    plt.title("Árvore de Decisão - Credit Data", fontsize=20)
    plt.show()

def algoritmo_random_forest():
    # abrir o arquivo
    with open('credit.pkl', 'rb') as f:
        X_credit_treinamento, y_credit_treinamento, X_credit_teste, y_credit_teste = pickle.load(f)

    """
    b) Para gerar a classificação você deve adicionar alguns parâmetros:
*   n_estimators=10  --> número de árvores que você irá criar
*   criterion='entropy'
*   random_state = 0

    """
    random_forest = RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=0)
    random_forest.fit(X_credit_treinamento, y_credit_treinamento)

    """
    c) Faça a previsão com os dados de teste. Visualize os dados e verifique se as previsões estão de acordo com os dados de teste (respostas reais).
    """
    previsoes = random_forest.predict(X_credit_teste)
    
    print("\nComparação entre previsões e valores reais (10 primeiras amostras):")
    for i in range(10):
        print(f"Amostra {i+1}: Previsto = {previsoes[i]}, Real = {y_credit_teste[i]}")

    """
    d) Agora faça o cálculo da acurácia para calcular a taxa de acerto entre os valores reais (y teste) e as previsões. 
    O resultado foi melhor do que a árvore de decisão simples?
    """
    acuracia = accuracy_score(y_credit_teste, previsoes)
    print(f"\nAcurácia do Random Forest: {acuracia:.2%}")
    
    # Comparação com a árvore de decisão simples (do exercício anterior)
    arvore_credit = DecisionTreeClassifier(random_state=0)
    arvore_credit.fit(X_credit_treinamento, y_credit_treinamento)
    acuracia_arvore = accuracy_score(y_credit_teste, arvore_credit.predict(X_credit_teste))
    print(f"Acurácia da Árvore de Decisão Simples: {acuracia_arvore:.2%}")
    
    if acuracia > acuracia_arvore:
        print("O Random Forest obteve melhor resultado que a árvore de decisão simples.")
    elif acuracia < acuracia_arvore:
        print("O Random Forest obteve resultado inferior à árvore de decisão simples.")
    else:
        print("O Random Forest obteve o mesmo resultado que a árvore de decisão simples.")

    """
    e) Se o resultado foi inferior, como você poderia resolver isso? Quais foram os resultados obtidos?

    Possíveis melhorias
    1. Aumentar o número de árvores (n_estimators)
    2. Ajustar outros parâmetros como max_depth, min_samples_split, etc.
    3. Utilizar cross-validation para encontrar melhores parâmetros
    4. Verificar balanceamento das classes e possivelmente aplicar técnicas como oversampling
    
    f) Faça a análise da Matriz de Confusão.
    """
    print("\nMatriz de Confusão:")
    cm = ConfusionMatrix(random_forest)
    cm.fit(X_credit_treinamento, y_credit_treinamento)
    cm.score(X_credit_teste, y_credit_teste)
    plt.title("Matriz de Confusão - Random Forest")
    plt.show()
    
    # Extraindo valores da matriz de confusão para análise
    from sklearn.metrics import confusion_matrix
    matriz = confusion_matrix(y_credit_teste, previsoes)
    print("\nAnálise da Matriz de Confusão:")
    print(f"Verdadeiros Positivos (TP): {matriz[1][1]}")
    print(f"Falsos Positivos (FP): {matriz[0][1]}")
    print(f"Verdadeiros Negativos (TN): {matriz[0][0]}")
    print(f"Falsos Negativos (FN): {matriz[1][0]}")

    """
    g) Faça um print do parâmetro classification_report entre os dados de teste e as previsões. 
    Explique qual é a relação entre precision e recall nos dados. Como você interpreta esses dados?
    """
    print("\nClassification Report:")
    report = classification_report(y_credit_teste, previsoes)
    print(report)
    
    """
    Interpretação:
    Precision (precisão): Proporção de previsões positivas que foram corretas.
    Recall (revocação): Proporção de casos positivos reais que foram identificados.
    F1-score: Média harmônica entre precision e recall, útil quando há desbalanceamento de classes.
    No contexto de crédito:
    - Alta precision para classe 0 (pagadores) significa que quando classificamos como pagador, temos alta confiança.
    - Alto recall para classe 1 (não pagadores) significa que estamos capturando a maioria dos maus pagadores.
    Geralmente há um trade-off entre precision e recall que deve ser balanceado conforme o risco que o banco está disposto a assumir.
    
    """

def main():
    algoritmo_arvore_decisao_menor()
    algoritmo_arvore_decisao_maior()
    algoritmo_random_forest()

if __name__ == "__main__":
    main()