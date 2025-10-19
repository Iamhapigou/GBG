import ujson
import pandas as pd
import numpy as np
import os

def convert_1():
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    filepath = os.path.dirname(os.path.abspath(__file__))
    # Convert the json file to csv
    with open(os.path.join(filepath, "dataset", "MNIST", "config.json"), "r") as f:
        data = ujson.load(f)

    data_1 = data["Size of samples for labels in clients"]
    data_2 = []

    for j in range(len(data_1)):
        for i in range(data["num_classes"]):
            data_2.append(j)
    data_3 = np.zeros((data["num_clients"], data["num_classes"], 2), dtype=int)

    for i in range(data["num_clients"]):
        x = np.array(data_1[i])[:,0]
        for j in range(data["num_classes"]):
            if j in x:
                idx = np.where(j==x)
                data_3[i,j] = np.array(data_1[i])[idx]
            else:
                data_3[i,j] = np.array([j,0])

    col_1 = np.reshape(np.array(data_2), (300,1))
    col_2 = np.reshape(data_3[:, :, 0], (300, 1))
    col_3 = np.reshape(data_3[:, :, 1], (300, 1))

    col = np.concatenate([col_1, col_2, col_3], axis=1)

    df = pd.DataFrame(col)
    print(df)

    df.to_csv("1_30_MNIST_data.csv", index=False)

def convert_2():
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    # Convert the json file to csv
    filepath = os.path.dirname(os.path.abspath(__file__))

    with open(os.path.join(filepath, "MNIST_result.json"), "r") as f:
        data_1 = ujson.load(f)
    with open(os.path.join(filepath, "dataset", "MNIST", "config.json"), "r") as f:
        data_2 = ujson.load(f)

    data1 = []
    data2 = []
    data_2 = data_2["Size of samples for labels in clients"]
    data3 = np.zeros((300, 1), dtype=int)

    for i in range(len(data_1)):
        for j in data_1[i]:
            for _ in range(10):
                data1.append(j)

    for i in range(30):
        for j in range(10):
            data2.append(j)

    print(data_1)
    flattened = [item for sublist in data_1 for item in sublist]

    for h, i in enumerate(np.array(flattened)):
        x = np.array(data_2[i])[:,0]
        count = -1
        for j in range(10):
            if j in x:
                count += 1
                data3[(h-1)*10+j, 0] = np.array(data_2[i])[count, 1]
            else:
                data3[(h-1)*10+j, 0] = np.array(0)

    col_1 = np.reshape(data1, (300, 1))
    col_2 = np.reshape(data2, (300, 1))
    col_3 = np.reshape(data3, (300, 1))

    col = np.concatenate([col_1, col_2, col_3], axis=1)

    df = pd.DataFrame(col)

    df.to_csv("1_30_MNIST_GROUP_data.csv", index=False)

convert_1()
convert_2()