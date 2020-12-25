import csv
import datetime
import json
import re

import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

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
path_to_json = '/Users/shanmukavarma/Downloads/anonymized/'
json_files = [pos_json for pos_json in os.listdir(path_to_json) if pos_json.endswith('.json')]
bigram = []
filenames = []


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


def get_tf_idf_cosine_similarity(compare_doc,doc_corpus):
    tf_idf_vect = TfidfVectorizer(stop_words=None, use_idf=True, norm=None)
    tf_idf_req_vector = tf_idf_vect.fit_transform([compare_doc]).todense()
    tf_idf_resume_vector = tf_idf_vect.transform(doc_corpus).todense()
    cosine_similarity_list = []
    for i in range(len(tf_idf_resume_vector)):
        cosine_similarity_list.append(cosine_similarity(tf_idf_req_vector,tf_idf_resume_vector[i])[0][0])
    return cosine_similarity_list


def process_files(req_document,resume_docs,filenames):
    # TF-IDF - cosine similarity
    cos_sim_list = get_tf_idf_cosine_similarity(req_document,resume_docs)
    zipped_docs = zip(cos_sim_list,filenames)
    sorted_doc_list = sorted(zipped_docs, key = lambda x: x[0], reverse=True)
    return sorted_doc_list


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
    f.close()
    if (len(json_content) > 0):
        contents.append(str(preprocess(json_content)))
        filenames.append(str(file_name[1]))
print('loading and pre-processing completed '+str(datetime.datetime.now()))
reg_document_text = 'Responsible for execution of various corporate finance deals and projects especially in merger & acquisitions area to ensure clients requests and regulatory requirements are met Assist in developing corporate finance business strategy in accordance with overall company strategy and contribute to the implementation of such business plans Maintaining good working relationship with clients and professional parties to ensure smooth execution of business projects Prepare marketing and presentation materials Frequent travel is required At least 5 years relevant experience in investment banks Strong command of spoken and written English and Chinese including both Cantonese & Mandarin is essential'
jd_data= preprocess(reg_document_text)
resultData = process_files(jd_data,contents,filenames)
print('JD Score completed '+str(datetime.datetime.now()))
with open('/Users/shanmukavarma/Downloads/jd-score.csv','wb') as out:
    csv_out=csv.writer(out)
    csv_out.writerow(['score','file_name'])
    for row in resultData:
        csv_out.writerow(row)