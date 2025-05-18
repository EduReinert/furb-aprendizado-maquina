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

def carregar_dados_pkl():
    #1. Utilize a base de dados construída no Trabalho 3 ‘risco_credito.pkl’, que possui 14 registros, para testar o algoritmo de Regressão Logística.
    arquivo_encoded = 'risco_credito.pkl'

    with open(arquivo_encoded, 'rb') as f:
        dados = pickle.load(f)

    X, y = dados
    
    #2. Faça o Encoder dos dados e, para facilitar, como fizemos na aula teórica, apague os registros que possuem a classe ‘moderado’. No total teremos 11 registros.
    mascara = y != 'moderado'
    
    X_filtrado = X[mascara]
    y_filtrado = y[mascara]
    
    return X_filtrado, y_filtrado

def main():
    """
    Regressão Logística
    """
    X_risco_credito, y_risco_credito = carregar_dados_pkl()
    print (len(X_risco_credito))

if __name__ == "__main__":
    main()