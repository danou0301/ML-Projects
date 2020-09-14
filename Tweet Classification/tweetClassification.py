from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from nltk.tokenize import RegexpTokenizer
from sklearn.naive_bayes import MultinomialNB
from language_detector import detect_language
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from keras.preprocessing.sequence import pad_sequences
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import RandomizedSearchCV


from keras.models import Sequential
from keras.preprocessing.text import Tokenizer

from keras import layers


from sklearn import metrics
import pandas
import os
import glob
import pandas as pd

def combine_all_files(dir):
    os.chdir("//Users/idan/Desktop/iml-tweet/tweetClassification")
    extension = 'csv'
    all_filenames = [i for i in glob.glob('*.{}'.format(extension))]
    #combine all files in the list
    combined_csv = pd.concat([pd.read_csv(f) for f in all_filenames ])
    #export to csv
    combined_csv.to_csv("All.csv", index=False, encoding='utf-8-sig')



def remove_link(): # n'ameliore pas
    data['tweet'] = data.tweet.str.replace('http', '')

def remove_non_alphaNumeric():
    data.tweet.str.replace('\W', '')


data = pandas.read_csv('All.csv')
#print(data.tweet)
#print(data.tweet.tolist())

def try2():
    tf = TfidfVectorizer()
    text_tf = tf.fit_transform(data['tweet'])
    X_train, X_test, y_train, y_test = train_test_split(
        text_tf, data['user'], test_size=0.3, random_state=123)
    clf = MultinomialNB().fit(X_train, y_train)
    predicted= clf.predict(X_test)
    print("MultinomialNB Accuracy:", metrics.accuracy_score(y_test, predicted))

def try1():  # best 0.784 avec test-size= 0.28
    token = RegexpTokenizer(r'[a-zA-Z0-9]+')
    cv = CountVectorizer(lowercase=True, stop_words='english', ngram_range=(1, 1), tokenizer=token.tokenize)
    text_counts = cv.fit_transform(data['tweet'])
    X_train, X_test, y_train, y_test = train_test_split(text_counts, data['user'], test_size=0.28, random_state=1000)
    clf = MultinomialNB().fit(X_train, y_train)
    predicted= clf.predict(X_test)

    realList = y_test.tolist()
    predictList = predicted.tolist()
    noGood = [0,0,0,0,0,0,0,0,0,0]
    for i in range(len(y_test)):

        if (realList[i] != predictList[i]):
            noGood[realList[i]] += 1

    print(noGood)
    print("MultinomialNB Accuracy:", metrics.accuracy_score(y_test, predicted))



# https://realpython.com/python-keras-text-classification/
def deep1():  # bof
    tweet_train, tweet_test, y_train, y_test = train_test_split(data['tweet'].values, data['user'].values, test_size=0.28, random_state=1000)
    vectorizer = CountVectorizer()
    vectorizer.fit(tweet_train)
    X_train = vectorizer.transform(tweet_train)
    X_test = vectorizer.transform(tweet_test)
    input_dim = X_train.shape[1]  # Number of features
    model = Sequential()
    model.add(layers.Dense(10, input_dim=input_dim, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    model.summary()
    history = model.fit(X_train, y_train,
                        epochs=10,
                        verbose=False,
                        validation_data=(X_test, y_test),
                        batch_size=10)

    loss, accuracy = model.evaluate(X_train, y_train, verbose=False)
    print("Training Accuracy: {:.4f}".format(accuracy))
    loss, accuracy = model.evaluate(X_test, y_test, verbose=False)
    print("Testing Accuracy:  {:.4f}".format(accuracy))

def deep2():
    tweet_train, tweet_test, y_train, y_test = train_test_split(data['tweet'].values, data['user'].values,
                                                                test_size=0.28, random_state=1000)

    embedding_dim = 50
    maxlen = 150
    tokenizer = Tokenizer(num_words=5000)
    tokenizer.fit_on_texts(tweet_train)

    X_train = tokenizer.texts_to_sequences(tweet_train)
    X_test = tokenizer.texts_to_sequences(tweet_test)

    vocab_size = len(tokenizer.word_index) + 1

    model = Sequential()
    model.add(layers.Embedding(input_dim=vocab_size,
                               output_dim=embedding_dim,
                               input_length=maxlen))
    model.add(layers.GlobalMaxPool1D())
    model.add(layers.Dense(10, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    model.summary()
    history = model.fit(X_train, y_train,
                        epochs=50,
                        verbose=False,
                        validation_data=(X_test, y_test),
                        batch_size=10)

    loss, accuracy = model.evaluate(X_train, y_train, verbose=False)
    print("Training Accuracy: {:.4f}".format(accuracy))
    loss, accuracy = model.evaluate(X_test, y_test, verbose=False)
    print("Testing Accuracy:  {:.4f}".format(accuracy))

#deep2()

def deep3(): #Convolutional Neural Networks

    tweet_train, tweet_test, y_train, y_test = train_test_split(data['tweet'].values, data['user'].values,
                                                                test_size=0.28, random_state=1000)

    embedding_dim = 100
    maxlen = 150
    tokenizer = Tokenizer(num_words=5000)
    tokenizer.fit_on_texts(tweet_train)

    X_train = tokenizer.texts_to_sequences(tweet_train)
    X_test = tokenizer.texts_to_sequences(tweet_test)

    vocab_size = len(tokenizer.word_index) + 1

    model = Sequential()
    model.add(layers.Embedding(vocab_size, embedding_dim, input_length=maxlen))
    model.add(layers.Conv1D(128, 5, activation='relu'))
    model.add(layers.GlobalMaxPooling1D())
    model.add(layers.Dense(10, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    model.summary()
    history = model.fit(X_train, y_train,
                        epochs=10,
                        verbose=False,
                        validation_data=(X_test, y_test),
                        batch_size=10)

    loss, accuracy = model.evaluate(X_train, y_train, verbose=False)
    print("Training Accuracy: {:.4f}".format(accuracy))
    loss, accuracy = model.evaluate(X_test, y_test, verbose=False)
    print("Testing Accuracy:  {:.4f}".format(accuracy))


def create_model(num_filters, kernel_size, vocab_size, embedding_dim, maxlen):
    model = Sequential()
    model.add(layers.Embedding(vocab_size, embedding_dim, input_length=maxlen))
    model.add(layers.Conv1D(num_filters, kernel_size, activation='relu'))
    model.add(layers.GlobalMaxPooling1D())
    model.add(layers.Dense(10, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

def deep4():  # Hyperparameters Optimization
    epochs = 20
    embedding_dim = 50
    maxlen = 100
    tweet_train, tweet_test, y_train, y_test = train_test_split(data['tweet'].values, data['user'].values,
                                                                test_size=0.28, random_state=1000)
    # Tokenize words
    tokenizer = Tokenizer(num_words=5000)
    tokenizer.fit_on_texts(tweet_train)
    X_train = tokenizer.texts_to_sequences(tweet_train)
    X_test = tokenizer.texts_to_sequences(tweet_test)
    vocab_size = len(tokenizer.word_index) + 1

    X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
    X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)

    param_grid = dict(num_filters=[32, 64, 128],
                      kernel_size=[3, 5, 7],
                      vocab_size=[vocab_size],
                      embedding_dim=[embedding_dim],
                      maxlen=[maxlen])
    model = KerasClassifier(build_fn=create_model,
                            epochs=epochs, batch_size=10,
                            verbose=False)
    grid = RandomizedSearchCV(estimator=model, param_distributions=param_grid,
                              cv=4, verbose=1, n_iter=5)
    grid_result = grid.fit(X_train, y_train)

    # Evaluate testing set
    test_accuracy = grid.score(X_test, y_test)
    print(test_accuracy)

deep4()