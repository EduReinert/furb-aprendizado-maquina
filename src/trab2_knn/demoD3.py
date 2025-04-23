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

def grupo_de_dados_2():
    mat = scipy.loadmat('grupoDados2.mat')

    grupoTrain = mat['grupoTrain']
    trainRots = mat['trainRots']
    grupoTest = mat['grupoTest']
    testRots = mat['testRots']

    resultado = testa_valores_de_k(grupoTrain, trainRots, grupoTest, testRots)
    melhor_k, melhor_acuracia = resultado["melhor_k"], resultado["melhor_acuracia"]

    # print("--------")
    # for i in grupoTrain:
    #     for num in i:
    #         print(f"{num:.2f}, ", end="")
    #     print("")
    # print("--------")

    # for i in grupoTest:
    #     for num in i:
    #         print(f"{num:.2f}, ", end="")
    #     print("")

    # Q2.1: Aplique seu kNN a este problema. Qual é a sua acurácia de classificação?
    print(f"k de maior acurácia: {melhor_k} ({melhor_acuracia})")

    # min_max_list = [
    #     {'min': np.min(grupoTrain[:, col]), 'max': np.max(grupoTrain[:, col])} for col in range(grupoTrain.shape[1])
    # ]

    grupoTrain = (grupoTrain - grupoTrain.min(axis=0)) / (grupoTrain.max(axis=0) - grupoTrain.min(axis=0))
    grupoTest = (grupoTest - grupoTest.min(axis=0)) / (grupoTest.max(axis=0) - grupoTest.min(axis=0))

    # Initialize an empty list to store the normalized test data
    # test_data_normalized = []

    # Loop through each row of the test data
    # for row in grupoTest:
    #     normalized_row = []

    #     # Loop through each column value in the row
    #     for col, value in enumerate(row):
    #         # Get the min and max for the current column from the training data
    #         column_min = min_max_list[col]['min']
    #         column_max = min_max_list[col]['max']

    #         # Apply the Min-Max scaling formula
    #         normalized_value = (value - column_min) / (column_max - column_min)

    #         # Append the normalized value to the normalized row
    #         normalized_row.append(normalized_value)

    #     # Append the normalized row to the list of normalized test data
    #     test_data_normalized.append(normalized_row)

    resultado = testa_valores_de_k(grupoTrain, trainRots, grupoTest, testRots)
    melhor_k, melhor_acuracia = resultado["melhor_k"], resultado["melhor_acuracia"]
    # print("--------")
    #
    # for i in grupoTrain:
    #     for num in i:
    #         print(f"{num:.2f}, ", end="")
    #     print("")
    # print("--------")
    #
    # for i in grupoTest:
    #     for num in i:
    #         print(f"{num:.2f}, ", end="")
    #     print("")

    # Q2.1: Aplique seu kNN a este problema. Qual é a sua acurácia de classificação?
    print(f"k de maior acurácia: {melhor_k} ({melhor_acuracia})")

    print("Q2.2: A acurácia pode ser igual a 98% com o kNN. Descubra por que o resultado atual é muito menor. Ajuste o conjunto de dados ou k de tal forma que a acurácia se torne 98% e explique o que você fez e por quê.")
    print("A acurácia na primeira tentativa foi menor porque os dados não estavam normalizados.")



def main():

    grupo_de_dados_2()

if __name__ == "__main__":
    main()