import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import pickle
from sklearn.linear_model import LogisticRegression

"""
ALUNOS:

Augusto Arraga
Eduardo Reinert
Vinicius Vanelli
"""

def regressao_logistica():
    #1. Utilize a base de dados construída no Trabalho 3 ‘risco_credito.pkl’, que possui 14 registros, para testar o algoritmo de Regressão Logística.
    arquivo_encoded = 'risco_credito.pkl'

    with open(arquivo_encoded, 'rb') as f:
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

def main():
    # PARTE 3: Regressão Logística (risco_credito.pkl)
    regressao_logistica('risco_credito.pkl')
    
    
    
if __name__ == "__main__":
    main()