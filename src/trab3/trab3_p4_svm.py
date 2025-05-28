import pandas as pd
import numpy as np
import pickle
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV

def svm_credit():
    # Carregar os dados
    with open('credit.pkl', 'rb') as f:
        X_credit_treinamento, y_credit_treinamento, X_credit_teste, y_credit_teste = pickle.load(f)
    
    # 1 e 2. SVM com kernel linear, C=1.0 e random_state=1
    print("\n=== SVM com Kernel Linear ===")
    svm_linear = SVC(kernel='linear', C=1.0, random_state=1)
    svm_linear.fit(X_credit_treinamento, y_credit_treinamento)
    previsoes_linear = svm_linear.predict(X_credit_teste)
    
    # 3. Utilize o comando do sklearn accuray_score para calcular a acurácia do seu algoritmo
    acuracia_linear = accuracy_score(y_credit_teste, previsoes_linear)
    print(f"Acurácia (kernel linear): {acuracia_linear:.3f}")
    
    # 4. Teste os demais kernels e anote os resultados. Qual o melhor kernel para a sua base de dados?
    kernels = ['poly', 'sigmoid', 'rbf']
    resultados_kernels = {}
    
    for kernel in kernels:
        svm = SVC(kernel=kernel, C=1.0, random_state=1)
        svm.fit(X_credit_treinamento, y_credit_treinamento)
        previsoes = svm.predict(X_credit_teste)
        acuracia = accuracy_score(y_credit_teste, previsoes)
        resultados_kernels[kernel] = acuracia
        print(f"Acurácia (kernel {kernel}): {acuracia:.3f}")
    
    melhor_kernel = max(resultados_kernels, key=resultados_kernels.get)
    # Melhor kernel: rbf com acurácia 0.982
    print(f"\nMelhor kernel: {melhor_kernel} com acurácia {resultados_kernels[melhor_kernel]:.3f}")
    
    # 5. Aumente o valor do parâmetro C aplicado ao melhor kernel e verifique se há mudanças no resultado do seu SVM.
    print("\n=== Ajuste do parâmetro C ===")
    for c_value in [0.1, 1.0, 10.0, 100.0]:
        svm = SVC(kernel=melhor_kernel, C=c_value, random_state=1)
        svm.fit(X_credit_treinamento, y_credit_treinamento)
        previsoes = svm.predict(X_credit_teste)
        acuracia = accuracy_score(y_credit_teste, previsoes)
        print(f"C = {c_value}: Acurácia = {acuracia:.3f}")
    
    # Melhorou. Com C = 100.0, a Acurácia foi de 0.990

    # 6. Usar Grid Search para encontrar melhores hiperparâmetros
    parametros = {
        'C': [0.001, 0.01, 0.1, 1, 5, 10, 50, 100, 500, 1000],
        'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
        'gamma': ['scale', 'auto']
    }
    
    grid_search = GridSearchCV(estimator=SVC(random_state=1), 
                             param_grid=parametros,
                             scoring='accuracy',
                             cv=5,
                             n_jobs=-1)
    
    grid_search.fit(X_credit_treinamento, y_credit_treinamento)
    
    # Melhores parâmetros encontrados
    print("\n=== Grid Search ===")
    # Melhores parâmetros: {'C': 1000, 'gamma': 'scale', 'kernel': 'rbf'}
    print(f"Melhores parâmetros: {grid_search.best_params_}")
    # Melhor acurácia (GridSearch): 0.993
    print(f"Melhor acurácia (GridSearch): {grid_search.best_score_:.3f}")
    
    # Avaliar no conjunto de teste
    melhor_svm = grid_search.best_estimator_
    previsoes_grid = melhor_svm.predict(X_credit_teste)
    acuracia_grid = accuracy_score(y_credit_teste, previsoes_grid)
    # Acurácia no teste (GridSearch): 0.988 
    print(f"Acurácia no teste (GridSearch): {acuracia_grid:.3f}")

    # O modelo com melhor desempenho foi obtido com os parâmetros ajustados manualmente ou com o GridSearch?
    # A acurácia com os parâmetros ajustados manualmente foi um pouco maior do que a obtida com o GridSearch com 5 Cross-Validation Folds (0.990 x 0.988)
    print("\n=== Comparação Final ===")
    print(f"Acurácia melhor modelo manual (kernel {melhor_kernel}, C=1.0): {resultados_kernels[melhor_kernel]:.3f}")
    print(f"Acurácia GridSearch: {acuracia_grid:.3f}")
    
    # Matriz de confusão e relatório de classificação para o melhor modelo
    print("\nMatriz de Confusão (GridSearch):")
    cmatrix = confusion_matrix(y_credit_teste, previsoes_grid)
    disp = ConfusionMatrixDisplay(confusion_matrix=cmatrix)
    disp.plot()
    plt.title('Matriz de Confusão - Melhor SVM')
    plt.show()
    
    print("\nClassification Report (GridSearch):")
    print(classification_report(y_credit_teste, previsoes_grid))

if __name__ == "__main__":
    svm_credit()


    """
6. O resultado do SVM é melhor que os resultados do Naive Bayes, Florestas Aleatórias e Regressão Logística? 
Descreva sua análise de resultados (observe que para isso você deverá visualizar os resultadosda Matriz de Confusão, acurácia, precisão e recall).

RESPOSTA:
O SVM, usando com kernel rbf e C alto, atingiu até 99% de acurácia, 
superando todos os outros algoritmos. Ele foi eficiente para as duas classes, 
mostrando o melhor desempenho geral entre os modelos testados.

    """