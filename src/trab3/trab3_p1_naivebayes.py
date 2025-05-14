import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.preprocessing import LabelEncoder
import pickle
from sklearn.naive_bayes import GaussianNB

def pre_processamento_dados(X_risco_credito, y_risco_credito):
    #Label Encoder
    label_encoder_historia = LabelEncoder()
    label_encoder_divida = LabelEncoder()
    label_encoder_garantias = LabelEncoder()
    label_encoder_renda = LabelEncoder()

    X_risco_credito[:, 0] = label_encoder_historia.fit_transform(X_risco_credito[:, 0])
    X_risco_credito[:, 1] = label_encoder_divida.fit_transform(X_risco_credito[:, 1])
    X_risco_credito[:, 2] = label_encoder_garantias.fit_transform(X_risco_credito[:, 2])
    X_risco_credito[:, 3] = label_encoder_renda.fit_transform(X_risco_credito[:, 3])

    with open('risco_credito.pkl', 'wb') as f:
        pickle.dump([X_risco_credito, y_risco_credito], f)

def naive_bayes(X_risco_credito, y_risco_credito):
    # Criar o objeto Nayve Bayes
    naiveb_risco_credito = GaussianNB()

    #Treinamento
    naiveb_risco_credito.fit(X_risco_credito, y_risco_credito)

    """
    Previsões
    """

    # Exemplo 1: História boa, dívida alta, garantia nenhuma, renda > 35
    exemplo_1 = [[0, 0, 1, 2]]  # Atributos de exemplo
    previsao_1 = naiveb_risco_credito.predict(exemplo_1)
    print(f'\nPrevisão, exemplo 1: {previsao_1[0]}')

    # Exemplo 2: História ruim, dívida alta, garantia adequada, renda < 15
    exemplo_2 = [[2, 0, 0, 0]]  # Atributos de exemplo
    previsao_2 = naiveb_risco_credito.predict(exemplo_2)
    print(f'Previsão, exemplo 2: {previsao_2[0]}')

    # utilize o atributo .classes_ para mostrar as classes utilizadas pelo algoritmo
    print("Classes detectadas pelo modelo:", naiveb_risco_credito.classes_)

    # utilize .class_count_ para contar quantos registros tem em cada classe
    print("Contagem de registros em cada classe:", naiveb_risco_credito.class_count_)


def main():
    #Entrada do dataset e transformação
    # a) DIVISÃO DA BASE DE DADOS
    dataset_risco_credito = pd.read_csv('dataset_risco_credito.csv')
    X_risco_credito = dataset_risco_credito.iloc[:, :-1].values  # Todas as colunas menos a última
    y_risco_credito = dataset_risco_credito.iloc[:, -1].values  # Apenas a última coluna

    #b) APLICAR LABEL ENCODER
    pre_processamento_dados(X_risco_credito, y_risco_credito)
    print('Após Label Encoder:')
    print(f'\nX_risco_credito', X_risco_credito)
    print(f'\ny_risco_credito', y_risco_credito)
    
    #chamada Naive Bayes
    naive_bayes(X_risco_credito, y_risco_credito)

if __name__ == "__main__":
    main()