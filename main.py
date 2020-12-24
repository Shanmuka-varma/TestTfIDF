import json
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from num2words import num2words
import os
import numpy as np
contents = []
path_to_json = '/Users/shanmukavarma/Downloads/anonymized/'
json_files = [pos_json for pos_json in os.listdir(path_to_json) if pos_json.endswith('.json')]


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
    symbols = "!\"#$%&()*+-./:;<=>?@[\]^_`{|}~\n\r\t"
    for i in range(len(symbols)):
        data = np.char.replace(data, symbols[i], ' ')
        data = np.char.replace(data, "  ", " ")
    data = np.char.replace(data, ',', '')
    data = np.char.replace(data, 'u2022', '')
    return data


def remove_apostrophe(data):
    return np.char.replace(data, "'", "")


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
        new_text = new_text + " " +lemmatizer.lemmatize(w)
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
    return data

for file_name in enumerate(json_files):
    total_file=path_to_json+file_name[1]
    f= open(total_file,)
    data = json.load(f)
    json_content = []
    if ((data is not None) and (data.has_key('EmployerOrg'))):
        for i in data['EmployerOrg']:
            if i.has_key('PositionHistory'):
                for z in i['PositionHistory']:
                    if z.has_key('Description'):
                        json_content.append(z['Description'])
    f.close();
    if(len(json_content) > 0) :
        contents.append(word_tokenize(str(preprocess(json_content))))
print('completed loadng files and pre-processing')
wordSet = set()
for i in contents:
    wordSet = wordSet.union(set(i))
tfData = []
wordDectList = []
Stringlist = list(wordSet)
f=open('/Users/shanmukavarma/Downloads/f1.txt','w')
for ele in Stringlist:
    f.write(ele+'\n')
f.close()
print('completed writing text for wordset')
def computeTF(wordDict, bow):
    tfDict = {}
    bowCount = len(bow)
    for word, count in wordDict.items():
        tfDict[word] = count/float(bowCount)
    return tfDict
for k in contents:
    wordDict = dict.fromkeys(wordSet, 0)
    for word in k:
        wordDict[word] += 1
    tfRow=computeTF(wordDict,k)
    tfData.append(tfRow)
    wordDectList.append(wordDict)
print('completed TF')
contents = []
def computeIDF(docList):
    import math
    idfDict = {}
    N = len(docList)

    idfDict = dict.fromkeys(docList[0].keys(), 0)
    for doc in docList:
        for word, val in doc.items():
            if val > 0:
                idfDict[word] += 1

    for word, val in idfDict.items():
        idfDict[word] = math.log10(N / float(val))

    return idfDict
idfs = computeIDF(wordDectList)
print('completed IDF')
def computeTFIDF(tfBow, idfs):
    tfidf = {}
    for word, val in tfBow.items():
        tfidf[word] = val*idfs[word]
    return tfidf
tfidfList = []
for i in tfData:
    settfidf = computeTFIDF(i,idfs)
    tfidfList.append(settfidf)
print('completed TFIDF')
import pandas as pd
df=pd.DataFrame(tfidfList)
df.to_csv('/Users/shanmukavarma/Downloads/test_file/result.csv')
