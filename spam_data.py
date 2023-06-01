import numpy as np
import pandas as pd
import chardet
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

with open('spam.csv', 'rb') as file:
    encoding = chardet.detect(file.read())['encoding']
    df_train = pd.read_csv('spam.csv', encoding=encoding, usecols=[0, 1])
label_encoder = LabelEncoder()

df_train['v1'] = label_encoder.fit_transform(df_train['v1'])
X_train = df_train['v2'].str.lower()

vec = CountVectorizer(stop_words=['end', 'on', 'is', 'of', 'as'])
X_train_transformed = vec.fit_transform(X_train)
y_train = df_train['v1'].astype(int)


with open('spam_test.csv', 'rb') as file:
    encoding = chardet.detect(file.read())['encoding']

    df_test = pd.read_csv('spam_test.csv', encoding=encoding, usecols=[0, 1])
df_test['v1'] = label_encoder.transform(df_test['v1'])
X_test = df_test['v2'].str.lower()
X_test_transformed = vec.transform(X_test)
y_test = df_test['v1'].astype(int)

mnb = MultinomialNB()
mnb.fit(X_train_transformed, y_train)

y_pred = mnb.predict(X_test_transformed)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
