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

def grupo_de_dados_1():
    mat = scipy.loadmat('grupoDados1.mat')

    grupoTrain = mat['grupoTrain']
    trainRots = mat['trainRots']
    grupoTest = mat['grupoTest']
    testRots = mat['testRots']

    # Previsto: 96%
    rotulo_previsto = knn.meuKnn(grupoTrain, trainRots, grupoTest, 1)
    print(f"Acurácia com k = 1: {knn.funcao_acuracia(rotulo_previsto, testRots):.2f}")

    # Previsto: 94%
    rotulo_previsto = knn.meuKnn(grupoTrain, trainRots, grupoTest, 10)
    print(f"Acurácia com k = 10: {knn.funcao_acuracia(rotulo_previsto, testRots):.2f}")

    # knn.visualizaPontos(grupoTest, testRots, 1, 2)

    resultado = testa_valores_de_k(grupoTrain, trainRots, grupoTest, testRots)
    melhor_k, melhor_acuracia = resultado["melhor_k"], resultado["melhor_acuracia"]

    # Q1.1. Qual é a acurácia máxima que você consegue da classificação?
    print(f"k de maior acurácia: {melhor_k} ({melhor_acuracia})")

    acuracias = []

    for i in range(0, len(grupoTrain[0])):
        _grupoTrain = remove_coluna(grupoTrain, i)
        _grupoTest = remove_coluna(grupoTest, i)

        acuracias.append({"coluna_removida": i,
                          "acuracia": knn.funcao_acuracia(knn.meuKnn(_grupoTrain, trainRots, _grupoTest, melhor_k),
                                                          testRots)})

    sorted_data = sorted(acuracias, key=lambda x: x["acuracia"], reverse=True)

    # Q1.2. É necessário ter todas as características (atributos) para obter a acurácia máxima para esta classificação?
    if (sorted_data[0]["acuracia"] > melhor_acuracia):
        print(
            f"Coluna removida pra melhorar acurácia: {sorted_data[0]['coluna_removida']} ({sorted_data[0]['acuracia']:.2f})")
    else:
        print("Remover colunas não melhora a acurácia.")

def main():
    grupo_de_dados_1()

if __name__ == "__main__":
    main()