import pickle
import pandas

from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import RegexpTokenizer


def load_pickle(filename):
    with open(filename, 'rb') as file:
        return pickle.load(file)


class Classifier:

    def classify(self, np_array):
        pandas_array = pandas.DataFrame(np_array)
        token = RegexpTokenizer(r'[a-zA-Z0-9]+')
        cv = CountVectorizer(lowercase=True, stop_words='english', ngram_range=(1, 1), tokenizer=token.tokenize,
                             vocabulary=pickle.load(open('feature.pkl', 'rb')))

        text_counts = cv.transform(pandas_array['tweet'])
        mainModel = load_pickle("mainModel.pkl")

        proba0 = mainModel.predict_proba(text_counts)[:, 0]
        proba1 = mainModel.predict_proba(text_counts)[:, 1]
        proba2 = mainModel.predict_proba(text_counts)[:, 2]
        proba3 = mainModel.predict_proba(text_counts)[:, 3]
        proba4 = mainModel.predict_proba(text_counts)[:, 4]
        proba5 = mainModel.predict_proba(text_counts)[:, 5]
        proba6 = mainModel.predict_proba(text_counts)[:, 6]
        proba7 = mainModel.predict_proba(text_counts)[:, 7]
        proba8 = mainModel.predict_proba(text_counts)[:, 8]
        proba9 = mainModel.predict_proba(text_counts)[:, 9]

        pandas_array['0'] = proba0
        pandas_array['1'] = proba1
        pandas_array['2'] = proba2
        pandas_array['3'] = proba3
        pandas_array['4'] = proba4
        pandas_array['5'] = proba5
        pandas_array['6'] = proba6
        pandas_array['7'] = proba7
        pandas_array['8'] = proba8
        pandas_array['9'] = proba9

        X = pandas_array.values[:, 1:10]

        probaModel = load_pickle("probaModel.pkl")
        y_pred = probaModel.predict(X)

        return y_pred.tolist()
