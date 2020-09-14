import numpy as np
import pandas as pnd
import matplotlib.pyplot as plt


def init_data():

    df = pnd.read_csv("kc_house_data.csv")
    df.dropna(how="any", inplace=True)  # If any NA values are present, drop that row or column.
                                        # inplace=True to delete the data not just return without the data
    # delete all no relevant datas
    df.drop(["date"], 1, inplace=True)

    # delete all corrupted datas
    df.drop(df[df["price"] <= 0].index, inplace=True)
    df.drop(df[df["bedrooms"]<=0].index, inplace=True)
    df.drop(df[df["bathrooms"]<=0].index, inplace=True)
    df.drop(df[df["sqft_living"]<=0].index, inplace=True)
    df.drop(df[df["sqft_lot"]<=0].index, inplace=True)
    df.drop(df[df["floors"]<=0].index, inplace=True)

    df.drop(df[df["view"]<0].index, inplace=True)
    df.drop(df[df["condition"]<0].index, inplace=True)
    df.drop(df[df["grade"]<0].index, inplace=True)
    df.drop(df[df["sqft_basement"]<0].index, inplace=True)

    df.insert(loc=0, column="intercept", value=[1 for i in range(len(df))])

    df = pnd.get_dummies(df, columns=["zipcode"])
    return df


def learn_data(data):
    train_errors = []
    tst_errors = []
    v = [i for i in range(1, 100)]

    for i in range(1, 100):
        train_sample = data.sample(frac=i/100)
        test_sample = data.drop(train_sample.index)

        train_result = train_sample["price"]
        test_result = test_sample["price"]

        train_sample.drop(["price"], 1, inplace=True)
        test_sample.drop(["price"], 1, inplace=True)

        data_matrix = pnd.DataFrame.as_matrix(train_sample)
        w = np.linalg.pinv(data_matrix)@train_result

        y_kova = data_matrix@w

        #testing
        train_errors.append(((np.linalg.norm(train_result-y_kova))**2)/len(train_result))
        test_matrix = pnd.DataFrame.as_matrix(test_sample)
        y_kova_test = np.dot(test_matrix, w)
        tst_errors.append(((np.linalg.norm(test_result - y_kova_test))**2)/len(test_result))

    plt.plot(v, train_errors, label="train mse")
    plt.plot(v, tst_errors, label="test mse")
    plt.legend()
    plt.title("training-testing set - test errors")
    plt.savefig("test-error.pdf")
    plt.show()

def corr(data):
    corr = data.corr()
    #print(corr)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
    fig.colorbar(cax)
    ticks = np.arange(0, len(data.columns), 1)
    ax.set_xticks(ticks)
    plt.xticks(rotation=90)
    ax.set_yticks(ticks)
    ax.set_xticklabels(data.columns)
    ax.set_yticklabels(data.columns)
    plt.show()

#corr(init_data())
learn_data(init_data())
