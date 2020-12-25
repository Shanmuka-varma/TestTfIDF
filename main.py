import json
import re

import nltk

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from num2words import num2words
import os
import numpy as np

contents = []
# path_to_json = '../target/anonymized/'
path_to_json = '/Users/shanmukavarma/Downloads/test_file/'
json_files = [pos_json for pos_json in os.listdir(path_to_json) if pos_json.endswith('.json')]
bigram = []


def convert_lower_case(data):
    return np.char.lower(data)


def remove_stop_words(data):
    stop_words = stopwords.words('english')
    words = word_tokenize(str(data))
    new_text = ""
    for w in words:
        if w not in stop_words and len(w) > 1:
            new_text = new_text + " " + w
    return new_text


def remove_symbols(data):
    symbols = "!\"#$%&()*+-./:;<=>?@[\]^_`{|}~\n\r"
    for i in range(len(symbols)):
        data = np.char.replace(data, symbols[i], ' ')
        data = np.char.replace(data, "  ", " ")
    data = np.char.replace(data, ',', '')
    data = np.char.replace(data, 'u2022', '')
    return data


def remove_apostrophe(data):
    return np.char.replace(data, "'", "")


def remove_numbers(data):
    return re.sub(r'\w*\d\w*', '', data).strip()


def stemming(data):
    stemmer = PorterStemmer()
    tokens = word_tokenize(str(data))
    new_text = ""
    for w in tokens:
        new_text = new_text + " " + stemmer.stem(w)
    return new_text


def lemmatize(data):
    lemmatizer = WordNetLemmatizer()
    tokens = word_tokenize(str(data))
    new_text = ""
    for w in tokens:
        new_text = new_text + " " + lemmatizer.lemmatize(w)
    return new_text


def remove_two_letter_words(data):
    tokens = word_tokenize(str(data))
    new_text = ""
    for w in tokens:
        if len(w) > 2:
            new_text = new_text + " " + w
    return new_text


def convert_numbers(data):
    tokens = word_tokenize(str(data))
    new_text = ""
    for w in tokens:
        try:
            w = num2words(int(w))
        except:
            a = 0
        new_text = new_text + " " + w
    new_text = np.char.replace(new_text, "-", " ")
    return new_text


def bigram_token(data):
    tokens = word_tokenize(str(data))
    bigrams_trigrams = list(map(' '.join, nltk.bigrams(tokens)))
    return bigrams_trigrams;


def preprocess(data):
    data = convert_lower_case(data)
    data = remove_symbols(data)
    data = remove_apostrophe(data)
    data = remove_stop_words(data)
    data = convert_numbers(data)
    data = lemmatize(data)
    data = remove_symbols(data)
    data = convert_numbers(data)
    data = lemmatize(data)  # needed again as we need to stem the words
    data = remove_symbols(data)  # needed again as num2word is giving few hypens and commas fourty-one
    data = remove_stop_words(data)  # needed again as num2word is giving stop words 101 - one hundred and one
    data = lemmatize(data)
    data = remove_numbers(data)
    data = remove_two_letter_words(data)
    tokens = bigram_token(data)
    bigram.append(tokens)
    return data


for file_name in enumerate(json_files):
    total_file = path_to_json + file_name[1]
    f = open(total_file, )

    try:
        data = json.load(f)
    except Exception:
        pass

    json_content = []
    if ((data is not None) and ('EmployerOrg' in data)):
        for i in data['EmployerOrg']:
            if 'PositionHistory' in i:
                for z in i['PositionHistory']:
                    if 'Description' in z:
                        json_content.append(z['Description'])
    f.close();
    if (len(json_content) > 0):
        contents.append(str(preprocess(json_content)))
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(max_df=.65, min_df=1, stop_words=None, use_idf=True, norm=None)
transformed_documents = vectorizer.fit_transform(contents)
print('vector')
transformed_documents_as_array = transformed_documents.toarray()
print(len(transformed_documents_as_array))
import pandas as pd
for counter, doc in enumerate(transformed_documents_as_array):
    # construct a dataframe
    tf_idf_tuples = list(zip(vectorizer.get_feature_names(), doc))
    one_doc_as_df = pd.DataFrame.from_records(tf_idf_tuples, columns=['term', 'score']).sort_values(by='score', ascending=False).reset_index(drop=True)

    # output to a csv using the enumerated value for the filename
    one_doc_as_df.to_csv('/Users/shanmukavarma/Downloads/'+str(counter)+'.csv')
