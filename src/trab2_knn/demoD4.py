import scipy.io as scipy
import implementacao_knn as knn

def maior_k_para_testar(value):
    percent = int(value * 0.2)  # 20% and truncate
    return max(1, min(percent, 20))

def remove_coluna(data, col_index):
    return [[x for i, x in enumerate(row) if i != col_index] for row in data]

def testa_valores_de_k(grupoTrain, trainRots, grupoTest, testRots):
    acuracias = []

    for i in range(0, maior_k_para_testar(len(grupoTrain))):
        acuracias.append({"k": i + 1,
                          "acuracia": knn.funcao_acuracia(knn.meuKnn(grupoTrain, trainRots, grupoTest, i + 1),
                                                          testRots)})

    sorted_data = sorted(acuracias, key=lambda x: x["acuracia"], reverse=True)

    return {"melhor_acuracia": sorted_data[0]['acuracia'], "melhor_k": sorted_data[0]['k']}

def grupo_de_dados_4():
    mat = scipy.loadmat('grupoDados4.mat')

    grupoTrain = mat['trainSet']
    trainRots = mat['trainLabs']
    grupoTest = mat['testSet']
    testRots = mat['testLabs']

    # Q4.1: Aplique seu kNN a este problema. Qual é a sua acurácia de classificação?
    #Normalização
    grupoTrain = (grupoTrain - grupoTrain.min(axis=0)) / (grupoTrain.max(axis=0) - grupoTrain.min(axis=0))
    grupoTest = (grupoTest - grupoTest.min(axis=0)) / (grupoTest.max(axis=0) - grupoTest.min(axis=0))

    #Testar valores de k para obter a melhor correspondência
    resultado = testa_valores_de_k(grupoTrain, trainRots, grupoTest, testRots)
    melhor_k, melhor_acuracia = resultado["melhor_k"], resultado["melhor_acuracia"]
    print(f"k de maior acurácia: {melhor_k} ({melhor_acuracia})")

    # Q4.2: A acurácia pode chegar a 92% com o K-NN. Descubra por que o resultado atual é muito menor. Ajuste o conjunto de dados ou o valor de k de forma que a acurácia atinja 92% e explique o que você fez e por quê. Observe que, desta vez, há mais de um problema...
    acuracias = []

    for i in range(0, len(grupoTrain[0])):
        _grupoTrain = remove_coluna(grupoTrain, i)
        _grupoTest = remove_coluna(grupoTest, i)

        acuracias.append({"coluna_removida": i,
                          "acuracia": knn.funcao_acuracia(knn.meuKnn(_grupoTrain, trainRots, _grupoTest, melhor_k),
                                                          testRots)})

    sorted_data = sorted(acuracias, key=lambda x: x["acuracia"], reverse=True)

    if (sorted_data[0]["acuracia"] > melhor_acuracia):
        print(
            f"Coluna removida pra melhorar acurácia: {sorted_data[0]['coluna_removida']} ({sorted_data[0]['acuracia']:.2f})")
    else:
        print("Remover colunas não melhora a acurácia.")

def main():
    grupo_de_dados_4()

if __name__ == "__main__":
    main()