import matplotlib.pyplot as plt
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from yellowbrick.classifier import ConfusionMatrix

"""
ALUNOS:

Augusto Arraga
Eduardo Reinert
Vinicius Vanelli
"""

def regressao_logistica():
    #1. Utilize a base de dados construída no Trabalho 3 ‘risco_credito.pkl’, que possui 14 registros, para testar o algoritmo de Regressão Logística.
    with open('risco_credito.pkl', 'rb') as f:
        dados = pickle.load(f)

    X, y = dados
    
    #2. Faça o Encoder dos dados e, para facilitar, como fizemos na aula teórica, apague os registros que possuem a classe ‘moderado’. No total teremos 11 registros.
    mascara = y != 'moderado'
    
    X_filtrado = X[mascara]
    y_filtrado = y[mascara]
    print(f'Total de registros: ', len(X_filtrado))
    
    #3. Treine o algoritmo de regressão logística e utilize o parâmetro ‘random_state =1’ para ter sempre o mesmo resultado.
    regressao_logistica = LogisticRegression(random_state=1)
    regressao_logistica = regressao_logistica.fit(X_filtrado, y_filtrado)
    
    #4. Utilize o comando ‘.intercept_’ para ter o resultado do B0.
    #O resultado deve ser =-0.80828993
    print(f'Resultado comando .intercept_: ', regressao_logistica.intercept_)
    
    #5. Utilize o comando ‘.coef_’ para ter o resultado dos demais parâmetros que deve ser:
    #array([[-0.76704533,  0.23906678, -0.47976059,  1.12186218]])
    print(f'Resultado comando .coef_: ', regressao_logistica.coef_)
    
    #6. Agora utilize o comando ‘predict’ para fazer o teste do seu algoritmo com:
    # a) história boa, dívida alta, garantias nenhuma, renda > 35 == BAIXO
    previsao_01 = regressao_logistica.predict([[0, 0, 1, 2]])
    print(f'Esperado: baixo \nObtido:', previsao_01)
    
    # b) história ruim, dívida alta, garantias adequada, renda < 15 == ALTO
    previsao_02 = regressao_logistica.predict([[2, 0, 0, 0]])
    print(f'Esperado: alto \nObtido:', previsao_02)

def regressao_logistica_maior():
    # abrir o arquivo
    with open('credit.pkl', 'rb') as f:
        X_credit_treinamento, y_credit_treinamento, X_credit_teste, y_credit_teste = pickle.load(f)
    
    #7. Agora aplique a Regressão Logística na base de dados ‘credit.pkl’. De quanto foi a taxa de acerto?
    regressao_logistica = LogisticRegression(random_state=1)
    regressao_logistica = regressao_logistica.fit(X_credit_treinamento, y_credit_treinamento)
    
    #Previsões
    previsoes = regressao_logistica.predict(X_credit_teste)
    
    #Taxa de acerto (acurácia)
    taxa_acerto = accuracy_score(y_credit_teste, previsoes)
    print(f'Taxa de acerto na base maior (credit.pkl): {taxa_acerto}')
    
    #Matriz de confusão
    cm = ConfusionMatrix(regressao_logistica)
    cm.fit(X_credit_treinamento, y_credit_treinamento)
    cm.score(X_credit_teste, y_credit_teste)
    plt.title('Matriz de Confusão')
    plt.show()

    # Classificação e recall
    print("\nClassification Report:")
    report = classification_report(y_credit_teste, previsoes)
    print(report)
    
    """
    8. O resultado com a base de dados ‘credit.pkl’ é melhor que os resultados do Naive Bayes e das Florestas Aleatórias? 
    Descreva sua análise de resultados (observe que para isso você deverá visualizar os resultados da Matriz de Confusão, acurácia, precisão e recall).
    
    RESPOSTA:
    A Regressão Logística teve acurácia de 94,6%, menor que a Árvore de Decisão (98%) 
    e Random Forest (96,8%). Ela foi boa para identificar pagadores, mas teve desempenho menor
    para inadimplentes. No geral, os métodos de árvore foram melhores nesse conjunto de dados.
    
    
    """

def main():
    # PARTE 3: Regressão Logística (risco_credito.pkl)
    regressao_logistica()
    print('\n\n\n')
    #Algoritmo de Regressão Logística para uma base de dados maior (Credit Data)
    regressao_logistica_maior()
    
    
    
if __name__ == "__main__":
    main()