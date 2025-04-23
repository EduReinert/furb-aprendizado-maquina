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

def grupo_de_dados_3():
    mat = scipy.loadmat('grupoDados3.mat')

    grupoTrain = mat['grupoTrain']
    trainRots = mat['trainRots']
    grupoTest = mat['grupoTest']
    testRots = mat['testRots']

    # Q2.1: Aplique seu kNN a este problema. Qual é a sua acurácia de classificação?
    rotulo_previsto = knn.meuKnn(grupoTrain, trainRots, grupoTest, 1)
    print(f"Acurácia com k = 1: {knn.funcao_acuracia(rotulo_previsto, testRots):.2f}")

    #Normalização
    grupoTrain = (grupoTrain - grupoTrain.min(axis=0)) / (grupoTrain.max(axis=0) - grupoTrain.min(axis=0))
    grupoTest = (grupoTest - grupoTest.min(axis=0)) / (grupoTest.max(axis=0) - grupoTest.min(axis=0))

    #Testar valores de k para obter a melhor correspondência
    resultado = testa_valores_de_k(grupoTrain, trainRots, grupoTest, testRots)
    melhor_k, melhor_acuracia = resultado["melhor_k"], resultado["melhor_acuracia"]
    print(f"k de maior acurácia: {melhor_k} ({melhor_acuracia})")

    print("Q3.2: A acurácia pode ser igual a 92% com o kNN. Descubra por que o resultado atual é muito menor. Ajuste o conjunto de dados ou k de tal forma que a acurácia se torne 92% e explique o que você fez e por quê.")
    print("A acurácia na primeira tentativa foi menor porque os dados não estavam normalizados. Além disso, testamos diversos valores de k para obter a melhor acurácia possível")


def main():

    grupo_de_dados_3()

if __name__ == "__main__":
    main()