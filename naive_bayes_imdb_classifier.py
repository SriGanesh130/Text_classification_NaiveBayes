import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn import metrics
from sklearn.metrics import roc_curve, auc, classification_report
from nltk.stem.porter import PorterStemmer
import re
from nltk.corpus import stopwords
import nltk
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score 

#Clearning data helper functions
def cleanHtml(sentence):
    cleanr = re.compile('<.*?>')
    cleantxt = re.sub(cleanr, ' ', sentence)
    return cleantxt

def cleanpunc(sentence):
    cleaned = re.sub(r'[?|!|\'|"|#]', r'', sentence)
    cleaned = re.sub(r'[.|,|)|(|\|/]', r' ', cleaned)
    return cleaned


def cleanCopus(corpus):
    sno = nltk.stem.SnowballStemmer("english")
    stop = set(stopwords.words("english"))
    all_positive_words = []
    all_negative_words = []
    final_string = []
    str1 = ''
    i = 0
    for string in corpus["review"].values:
        filtered_sentence = []
        # Removes html tags from every review
        sent = cleanHtml(string)
        for w in sent.split():
            # For every word in a review clean punctions
            for cleanwords in cleanpunc(w).split():
                # if cleaned is alphabet and length og words greater than 2 then proceed
                if ((cleanwords.isalpha()) and len(cleanwords)>2):
                    # check weather word is stop word or not
                    if cleanwords.lower() not in stop:
                        # If word is not stop word then append it to filtered sentence
                        s = (sno.stem(cleanwords.lower())).encode('utf-8')
                        filtered_sentence.append(s)
                        if (data["sentiment"].values)[i].lower() == "positive":
                            all_positive_words.append(s)
                        if (data["sentiment"].values)[i].lower() == "negative":
                            all_negative_words.append(s)
                    else:
                        continue
                else:
                    continue
        # filtered_sentence is list contains all words of a review after preprocessing
        # join every word in a list to get a string format of the review
        str1 = b" ".join(filtered_sentence)
        #append all the string(cleaned reviews)to final_string
        final_string.append(str1)
        i += 1      
    return final_string  

def word_embeddings(data):
    count_vect_tfidf = TfidfVectorizer(ngram_range = (1, 2))
    count_vect_tfidf = count_vect_tfidf.fit(data["review"].values)
    tfidf_wrds = count_vect_tfidf.transform(data["review"].values)
    return tfidf_wrds


def conv_label(label):
    if label.lower() == "positive":
        return 1
    elif label.lower() == "negative":
        return 0

#Read data
if __name__ == "__main__":
    data = pd.read_csv("./data/IMDB Dataset.csv")
    print(data["sentiment"].value_counts())
    #This will train only on 2000 data points, you can change it to 5000
    data_down = {"review" : data["review"][:2000], "sentiment" : data["sentiment"][:2000]}
    data_df = pd.DataFrame(data_down, columns = ["review", "sentiment"])
    print(data_df.head(5))
    print(data_df["review"][:10])
    data_df["review"] = cleanCopus(data_df)
    data_df["sentiment"] = data["sentiment"].map(conv_label)
    print(data_df["review"][:10])
    #TF-Idf vector using bi-grams
    tfidf_wrds = word_embeddings(data_df)
    X = tfidf_wrds
    Y = data_df["sentiment"]
    x_l, x_test, y_l, y_test = train_test_split(X, Y, test_size = 0.3, random_state = 42)
    from sklearn.naive_bayes import MultinomialNB
    clf = MultinomialNB(alpha = 0.7)
    clf.fit(x_l, y_l)
    pred = clf.predict(x_test)
    acc = accuracy_score(y_test, pred, normalize = True) * float(100)  
    print("acc is on test data:", acc)
    sns.heatmap(confusion_matrix(y_test, pred), annot = True, fmt = 'd')
    train_acc = accuracy_score(y_l, clf.predict(x_l), normalize = True) * float(100)
    print("train accuracy is:", train_acc)
    print(classification_report(y_test, pred))


