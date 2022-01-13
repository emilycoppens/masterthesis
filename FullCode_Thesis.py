#!/usr/bin/env python
# coding: utf-8

# # Packages

# In[ ]:


import pandas as pd
import numpy as np
import requests
import json
import csv 
import time
import datetime
import textblob  
from textblob import TextBlob
import string
import nltk
import ssl
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 
import re
import spacy
from nltk.corpus import sentiwordnet as swn
from IPython.display import clear_output
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import plotly
from nltk.corpus import wordnet as wn
from nltk.corpus import sentiwordnet as swn
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from collections import Counter
import matplotlib.gridspec as gridspec
from gridspec import GridSpec


# # Scraping Reddit

# In[ ]:


# Source
# <Rare Loot> (<28/10/2018>) <Using Pushshift’s API to extract Reddit Submissions> (<Version 1.0>) [<Source Code>], 
# https://rareloot.medium.com/using-pushshifts-api-to-extract-reddit-submissions-fb517b286563.

def PushshiftTrigger(query, after, before, sub):
    url = 'https://api.pushshift.io/reddit/search/submission/?title='+str(query)+'&size=1000&after='+str(after)+'&before='+str(before)+'&subreddit='+str(sub)
    r = requests.get(url)
    r.raise_for_status()
    if r.status_code != 204:
        data = json.loads(r.text)
        return data['data']
    
def SubmissionData(subm):
    subData = list()
    title = subm['title']
    submission = subm['selftext']
    url = subm['url']
    try:
        flair = subm['link_flair_text']
    except KeyError:
        flair = "NaN"
    
# Source   
# <djra> (<22/07/2014>) <Max retries exceeded with URL in requests> [<Source Code>],    
# https://stackoverflow.com/questions/23013220/max-retries-exceeded-with-url-in-requests
    except requests.ConnectionError as e:
        print("Connection Error.\n")
        print(str(e))            
        renewIPadress()
    except requests.Timeout as e:
        print("Timeout Error")
        print(str(e))
        renewIPadress()
    except requests.RequestException as e:
        print("General Error")
        print(str(e))
        renewIPadress()
    except KeyboardInterrupt:
        print("Program Interrupted")
    
    
    author = subm['author']
    sub_id = subm['id']
    score = subm['score']
    created = datetime.datetime.fromtimestamp(subm['created_utc']) 
    numComms = subm['num_comments']
    permalink = subm['permalink']
        
    subData.append((sub_id,submission,title,url,author,score,created,numComms,permalink,flair))
    subStats[sub_id] = subData


after = "1601986607" 
before = "1633522607"
query = "bitcoin"
sub = "investing"

subCount = 0
subStats = {}

data = PushshiftTrigger(query, after, before, sub)

while len(data) > 0: 
    for submission in data:
        SubmissionData(submission)
        subCount+=1
    print(len(data))
    print(str(datetime.datetime.fromtimestamp(data[-1]['created_utc'])))
    after = data[-1]['created_utc']
    data = PushshiftTrigger(query, after, before, sub)

def SubmissionToFile():
    upload_count = 0
    print("input filename of submission file, add .csv")
    filename = input() 
    file = filename
    with open(file, 'w', newline='', encoding='utf-8') as file: 
        a = csv.writer(file, delimiter=',')
        headers = ["Post ID","Submission","Title","Url","Author","Score","Publish Date","Total No. of Comments","Permalink","Flair"]
        a.writerow(headers)
        for sub in subStats:
            a.writerow(subStats[sub][0])
            upload_count+=1
            
        print(str(upload_count) + " submissions have been uploaded")
SubmissionToFile()


# In[ ]:


df1 = pd.read_csv('binance_altcoin.csv')
df2 = pd.read_csv('binance_crypto.csv')
df3 = pd.read_csv('binance_investing.csv')
df4 = pd.read_csv('binance_CryptoCurrencies.csv')
df5 = pd.read_csv('binance1_CryptoCurrency.csv')
df6 = pd.read_csv('binance2_CryptoCurrency.csv')
df7 = pd.read_csv('binance3_CryptoCurrency.csv')
df8 = pd.read_csv('binance_CryptoMarkets.csv')
df9 = pd.read_csv('binance_altcoin2.csv')
df10 = pd.read_csv('binance_crypto2.csv')
df11 = pd.read_csv('binance_investing2.csv')
df12 = pd.read_csv('binance_CryptoCurrencies2.csv')
df13 = pd.read_csv('binance_CryptoCurrency2.csv')
df14 = pd.read_csv('binance_CryptoMarkets2.csv')
binance = pd.concat((df1,df2,df3,df4,df5,df6,df7,df8,df9,df10,df11,df12,df13,df14), axis=0)
df1 = pd.read_csv('bitcoin_altcoin.csv')
df2 = pd.read_csv('bitcoin_crypto.csv')
df3 = pd.read_csv('bitcoin_investing.csv')
df4 = pd.read_csv('bitcoin1_CryptoCurrencies.csv')
df5 = pd.read_csv('bitcoin2_CryptoCurrencies.csv')
df6 = pd.read_csv('bitcoin3_CryptoCurrencies.csv')
df7 = pd.read_csv('bitcoin1_CryptoCurrency.csv')
df8 = pd.read_csv('bitcoin2_CryptoCurrency.csv')
df9 = pd.read_csv('bitcoin1_CryptoMarkets.csv')
df10 = pd.read_csv('bitcoin2_CryptoMarkets.csv')
df11 = pd.read_csv('bitcoin3_CryptoMarkets.csv')
df12 = pd.read_csv('bitcoin_altcoin2.csv')
df13 = pd.read_csv('bitcoin_crypto2.csv')
df14 = pd.read_csv('bitcoin_investing2.csv')
df15 = pd.read_csv('bitcoin_CryptoCurrencies2.csv')
df16 = pd.read_csv('bitcoin_CryptoCurrency2.csv')
df17 = pd.read_csv('bitcoin_CryptoMarkets2.csv')
bitcoin = pd.concat((df1,df2,df3,df4,df5,df6,df7,df8,df9,df10,df11,df12,df13,df14,df15,df16,df17), axis=0)
df1 = pd.read_csv('cardano_altcoin.csv')
df2 = pd.read_csv('cardano_crypto.csv')
df3 = pd.read_csv('cardano_investing.csv')
df4 = pd.read_csv('cardano_CryptoCurrencies.csv')
df5 = pd.read_csv('cardano_CryptoCurrency.csv')
df6 = pd.read_csv('cardano_CryptoMarkets.csv')
df7 = pd.read_csv('cardano_CryptoCurrencies2.csv')
df8 = pd.read_csv('cardano_CryptoCurrency2.csv')
df9 = pd.read_csv('cardano_CryptoMarkets2.csv')
cardano = pd.concat((df1,df2,df3,df4,df5,df6,df7,df8,df9), axis=0)
df1 = pd.read_csv('compound_altcoin.csv')
df2 = pd.read_csv('compound_crypto.csv')
df3 = pd.read_csv('compound_investing.csv')
df4 = pd.read_csv('compound_CryptoCurrencies.csv')
df5 = pd.read_csv('compound_CryptoCurrency.csv')
df6 = pd.read_csv('compound_CryptoMarkets.csv')
df7 = pd.read_csv('compound_crypto2.csv')
df8 = pd.read_csv('compound_investing2.csv')
df9 = pd.read_csv('compound_CryptoCurrencies2.csv')
df10 = pd.read_csv('compound_CryptoCurrency2.csv')
df11 = pd.read_csv('compound_CryptoMarkets2.csv')
compound = pd.concat((df1,df2,df3,df4,df5,df6,df7,df8,df9,df10,df11), axis=0)
df1 = pd.read_csv('dogecoin_altcoin.csv')
df2 = pd.read_csv('dogecoin_crypto.csv')
df3 = pd.read_csv('dogecoin_investing.csv')
df4 = pd.read_csv('dogecoin_CryptoCurrencies.csv')
df5 = pd.read_csv('dogecoin_CryptoCurrency.csv')
df6 = pd.read_csv('dogecoin_CryptoMarkets.csv')
df7 = pd.read_csv('dogecoin_altcoin2.csv')
df8 = pd.read_csv('dogecoin_investing2.csv')
df9 = pd.read_csv('dogecoin_CryptoCurrencies2.csv')
df10 = pd.read_csv('dogecoin_CryptoCurrency2.csv')
df11 = pd.read_csv('dogecoin_CryptoMarkets2.csv')
dogecoin = pd.concat((df1,df2,df3,df4,df5,df6,df7,df8,df9,df10,df11), axis=0)
df1 = pd.read_csv('solana_altcoin.csv')
df2 = pd.read_csv('solana_crypto.csv')
df3 = pd.read_csv('solana_investing.csv')
df4 = pd.read_csv('solana_CryptoCurrencies.csv')
df5 = pd.read_csv('solana_CryptoCurrency.csv')
df6 = pd.read_csv('solana_CryptoMarkets.csv')
df7 = pd.read_csv('solana_CryptoCurrencies2.csv')
df8 = pd.read_csv('solana_CryptoCurrency2.csv')
df9 = pd.read_csv('solana_CryptoMarkets2.csv')
solana = pd.concat((df1,df2,df3,df4,df5,df6,df7,df8,df9), axis=0)
df1 = pd.read_csv('safemoon_altcoin.csv')
df2 = pd.read_csv('safemoon_crypto.csv')
df3 = pd.read_csv('safemoon_investing.csv')
df4 = pd.read_csv('safemoon_CryptoCurrencies.csv')
df5 = pd.read_csv('safemoon1_CryptoCurrency.csv')
df6 = pd.read_csv('safemoon2_CryptoCurrency.csv')
df7 = pd.read_csv('safemoon_CryptoMarkets.csv')
safemoon = pd.concat((df1,df2,df3,df4,df5,df6, df7), axis = 0)
df1 = pd.read_csv('tether_altcoin.csv')
df2 = pd.read_csv('tether_crypto.csv')
df3 = pd.read_csv('tether_investing.csv')
df4 = pd.read_csv('tether_CryptoCurrencies.csv')
df5 = pd.read_csv('tether_CryptoCurrency.csv')
df6 = pd.read_csv('tether_CryptoMarkets.csv')
df7 = pd.read_csv('tether_crypto2.csv')
df8 = pd.read_csv('tether_CryptoCurrency2.csv')
df9 = pd.read_csv('tether_CryptoMarkets2.csv')
tether = pd.concat((df1,df2,df3,df4,df5,df6,df7,df8,df9), axis=0)

df = pd.concat((binance,bitcoin,cardano,compound,dogecoin,safemoon,solana,tether))
df['date'] = dates
df['week'] = pd.DatetimeIndex(df['date']).week
df['month'] = pd.DatetimeIndex(df['date']).month
df['year'] = pd.DatetimeIndex(df['date']).year
df['index'] = range(len(df))
df.to_csv(r'dataframe.csv')


# # Manual Annotation

# In[ ]:


manualannotation = fulldf.sample(n=100, random_state=1)
manualannotation.to_csv(r'manualannotation.csv')


# # Lexicons

# In[ ]:


df = pd.read_csv(r'dataframe.csv')
# Python program to convert a list to string: https://www.geeksforgeeks.org/python-program-to-convert-a-list-to-string/
    
def listtostring(data): 
    string = "" 
    for i in data: 
        string += "" 
        string += i
    return string


# In[ ]:



dftblob = df
lines_list = list()
for sentence in dftblob['Title']:
    print(listtostring(sentence))
    lines_list.append(listtostring(sentence))   
lineslist = []
for sentence in lines_list:
    sentence="".join([char for char in sentence if char not in string.punctuation])
    lineslist.append(sentence)
polaritylist = []
for text in lineslist:
    blob = TextBlob(text.strip("./,"))
    blob.tags
    blob.noun_phrases 
    for sentence in blob.sentences:
        polaritylist.append(sentence.sentiment.polarity)
dftblob['textblob_polarity'] = polaritylist

# Source   
# <Hamis Hisham> (<03/2021>) <Movies_Reviews_Sentiment_Analysis_Uptated> (Version 4.0) [<Source Code>],    
#https://www.kaggle.com/hamishisham/movies-reviews-sentiment-analysis-uptated
dfswn = df
def preprocessData(data,name):
    data[name]=data[name].str.lower()
    data[name]=data[name].apply(lambda x:re.sub(r'\B#\S+','',x))
    data[name]=data[name].apply(lambda x:re.sub(r"http\S+", "", x))
    data[name]=data[name].apply(lambda x:' '.join(re.findall(r'\w+', x)))
    data[name]=data[name].apply(lambda x:re.sub(r'\s+', ' ', x, flags=re.I))
    data[name]=data[name].apply(lambda x:re.sub(r'\s+[a-zA-Z]\s+', '', x))
    data[name]=data[name].apply(lambda x:re.sub('@[^\s]+','',x))

def stopwordsTokenization(data,name):
      
    def getting(sentence):
        example_sent = sentence
        
        filtered_sentence = [] 

        stop_words = set(stopwords.words('english')) 

        word_tokens = word_tokenize(example_sent) 
        
        filtered_sentence = [w for w in word_tokens if not w in stop_words] 
        
        return filtered_sentence
    x=[]
    for i in data[name].values:
        x.append(getting(i))
    data[name]=x

lemmatizer = WordNetLemmatizer()
def Lemmatization(data,name):
    def getting2(sentence):
        
        example = sentence
        output_sentence =[]
        word_tokens2 = word_tokenize(example)
        lemmatized_output = [lemmatizer.lemmatize(w) for w in word_tokens2]
        
        without_single_chr = [word for word in lemmatized_output if len(word) > 2]
        cleaned_data_title = [word for word in without_single_chr if not word.isnumeric()]
        
        return cleaned_data_title

def make_sentences(data,name):
    data[name]=data[name].apply(lambda x:' '.join([i+' ' for i in x]))
    data[name]=data[name].apply(lambda x:re.sub(r'\s+', ' ', x, flags=re.I))

preprocessData(dfswn,'Title')
stopwordsTokenization(dfswn,'Title')
make_sentences(dfswn,'Title') 

final_Edit = dfswn['Title'].copy()
dfswn['After_lemmatization'] = final_Edit

Lemmatization(dfswn,'After_lemmatization')
make_sentences(dfswn,'After_lemmatization')

pos=neg=obj=count=0

postagging = []

for review in dfswn['Title']:
    list = word_tokenize(review)
    postagging.append(nltk.pos_tag(list))

dfswn['pos_tags'] = postagging

def toWn(tag):
    if tag.startswith('J'):
        return wn.ADJ
    elif tag.startswith('N'):
        return wn.NOUN
    elif tag.startswith('R'):
        return wn.ADV
    elif tag.startswith('V'):
        return wn.VERB
    return None

def get_sentiment(word,tag):
    wn_tag = toWn(tag)
    
    if wn_tag not in (wn.NOUN, wn.ADJ, wn.ADV):
        return []

    lemma = lemmatizer.lemmatize(word, pos=wn_tag)
    if not lemma:
        return []

    synsets = wn.synsets(word, pos=wn_tag)
    if not synsets:
        return []

    synset = synsets[0]
    swn_synset = swn.senti_synset(synset.name())

    return [synset.name(), swn_synset.pos_score(),swn_synset.neg_score(),swn_synset.obj_score()]

    pos=neg=obj=count=0
    
senti_score = []

for pos_val in dfswn['pos_tags']:
    senti_val = [get_sentiment(x,y) for (x,y) in pos_val]
    for score in senti_val:
        try:
            pos = pos + score[1]
            neg = neg + score[2]
        except:
            continue
    senti_score.append(pos - neg)
    pos=neg=0    
    
dfswn['sentiwordnet_polarity'] = senti_score

dfvader = pd.read_csv(r'dataframe.csv')
dframe = []
data_frame = []
for i in dfvader['Title']:
    i = i.lower()
    i = "".join([char for char in i if char not in string.punctuation])
    i = word_tokenize(i)
    dframe.append(i)
lines_list = []
for sentence in dframe:
    lines_list.append(listtostring(sentence))
lines_list = []
for sentence in dfvader['Title']:
    lines_list.append(listtostring(sentence))
    compoundscores = []
for sentence in lines_list:
    sid = SentimentIntensityAnalyzer()
    print(sentence)
    ss = sid.polarity_scores(sentence)
    for k in sorted(ss):
        print('{0}: {1}, '.format(k, ss[k]), end='')
    compoundscores.append(ss['compound'])
    print()
dfvader['polarity'] = compoundscores


# # Polarities

# In[ ]:


mannotation = pd.read_csv("manualannotation_final.csv")

dfswn = pd.read_csv(r'sentiwordnet.csv')
index_list = []
for i in mannotation['index']:
    for index in dfswn['index']:
        if index == i:
            index_list.append(index)
roundedpolarity = list()
for i in dfswn['sentiwordnet_polarity']:
    if i > 0:
        roundedpolarity.append(1)
    elif i < 0:
        roundedpolarity.append(-1)
    else:
        roundedpolarity.append(0) 
        
dfswn['roundedpolarity'] = roundedpolarity
dfswn['roundedpolarity'].value_counts()

b = 0
bad = 0
good = 0
list_sentiwordnet = list()

for index in index_list:
    if dfswn['roundedpolarity'][index] == mannotation.iloc[b, 3]:
        list_sentiwordnet.append(dfswn['roundedpolarity'][index])
        good += 1
        b+=1
    elif dfswn['roundedpolarity'][index] != mannotation.iloc[b, 3]:
        list_sentiwordnet.append(dfswn['roundedpolarity'][index])
        bad +=1
        b +=1
print("Accuracy:", good/(good+bad))

dftblob = pd.read_csv(r'dftblob.csv')
textblob_roundedpolarity = list()
for i in dftblob['textblob_polarity']:
    if i > 0:
        textblob_roundedpolarity.append(1)
    elif i < 0:
        textblob_roundedpolarity.append(-1)
    else:
        textblob_roundedpolarity.append(0) 
        
dftblob['roundedpolarity'] = textblob_roundedpolarity
dftblob['roundedpolarity'].value_counts()
b = 0
bad = 0
good = 0
list_textblob = list()

for index in index_list:
    if dftblob['roundedpolarity'][index] == mannotation.iloc[b, 3]:
        list_textblob.append(dftblob['roundedpolarity'][index])
        good += 1
        b+=1
    elif dftblob['roundedpolarity'][index] != mannotation.iloc[b, 3]: 
        list_textblob.append(dftblob['roundedpolarity'][index])
        bad +=1
        b +=1
print("Accuracy:", good/(good+bad))

dfvader = pd.read_csv('dfvader.csv')
roundedpolarity = list()
for i in dfvader['polarity']:
    if i > 0:
        roundedpolarity.append(1)
    elif i < 0:
        roundedpolarity.append(-1)
    else:
        roundedpolarity.append(0) 
        
dfvader['roundedpolarity'] = roundedpolarity
dfvader['roundedpolarity'].value_counts()
b = 0
bad = 0
good = 0
list_vader = []

for index in index_list:
    if dfvader['roundedpolarity'][index] == mannotation.iloc[b, 3]:
        good += 1
        list_vader.append(dfvader['roundedpolarity'][index])
        b+=1
    elif dfvader['roundedpolarity'][index] != mannotation.iloc[b, 3]: 
        list_vader.append(dfvader['roundedpolarity'][index])
        bad +=1
        b +=1
print("Accuracy:", good/(good+bad))

# Source
# <gipsy> (<10/11/2015>) <Finding majority votes on -1s, 1s and 0s in list - python (<Version 1.0>) [<Source Code>], 
# https://stackoverflow.com/questions/33511259/finding-majority-votes-on-1s-1s-and-0s-in-list-python#comment54806213_33511352
def find_majority(votes):
    vote_count = Counter(votes)
    top_two = vote_count.most_common(2)
    if len(top_two)>1 and top_two[0][1] == top_two[1][1]:
        return 0
    return top_two[0][0]

templist = []
newlist = []
for i in range(len(dftblob['roundedpolarity'])):
        templist = [dftblob['roundedpolarity'][i], dfswn['roundedpolarity'][i], dfvader['roundedpolarity'][i]]
        newlist.append(find_majority(templist))
        b = 0
bad = 0
good = 0
list_majority = []

for index in index_list:
    if newlist[index] == mannotation.iloc[b, 3]:
        list_majority.append(newlist[index])
        good += 1
        b+=1
    elif newlist[index] != mannotation.iloc[b, 3]:         
        list_majority.append(newlist[index])
        bad +=1
        b +=1
print("Accuracy:", good/(good+bad))

confusion = confusion_matrix(mannotation['score'], list_vader)
print('Confusion Matrix: VADER\n')
print(confusion)
print('\nClassification Report\n')
print(classification_report(mannotation['score'], list_vader, target_names=['Class: -1', 'Class 0', 'Class 1']))

confusion = confusion_matrix(mannotation['score'], list_sentiwordnet)
print('Confusion Matrix: SentiWordNet\n')
print(confusion)
print('\nClassification Report\n')
print(classification_report(mannotation['score'], list_sentiwordnet, target_names=['Class: -1', 'Class 0', 'Class 1']))

confusion = confusion_matrix(mannotation['score'], list_textblob)
print('Confusion Matrix: TextBlob\n')
print(confusion)
print('\nClassification Report\n')
print(classification_report(mannotation['score'], list_textblob, target_names=['Class: -1', 'Class 0', 'Class 1']))

confusion = confusion_matrix(mannotation['score'], list_majority)
print('Confusion Matrix: Majority Vote\n')
print(confusion)
print('\nClassification Report\n')
print(classification_report(mannotation['score'], list_majority, target_names=['Class: -1', 'Class 0', 'Class 1']))

df = pd.read_csv(r'dataframe.csv')
df['polarity'] = dfvader['polarity']
df.to_csv(r'dataframe.csv')

binance = df[:15177]
bitcoin = df[15177:43706]
cardano = df[43706:45624]
compound = df[45624:46061]
dogecoin = df[46061:54239]
polygon = df[54239:56069]
safemoon = df[56069:59331]
solana = df[59331:61443]
tether = df[61443:63430]

plt.close("all")

coins = [binance,bitcoin,cardano,compound,dogecoin,polygon,safemoon,solana,tether]
coinsstr = ['binance','bitcoin','cardano','compound','dogecoin','polygon','safemoon','solana','tether']
b=0

for coin in coins:
    trial = coin[['date','polarity']]
    avg = trial.groupby('date')['polarity'].mean()
    setdate = list(set(coin['date']))
    setpolarity = set(coin['polarity'])
    newdf = pd.DataFrame()
    newdf['setdate'] = setdate
    newdf['avg'] = avg
    
    print(coinsstr[b])
    fig=  plt.figure(figsize=(30,8))
    ax = fig.add_subplot() 
    ax.plot(avg,label='average polarity') 
    ax.set_ylim([-1, 1])
    ax.set_xlabel('x date') 
    ax.set_ylabel('y polarity')
    ax.set_title('Polarity Graph') 
    ax.legend()
    plt.show()
    b+=1


# # Market Value

# In[ ]:


def datechanger(d):
    return datetime.strptime(d, '%b %d, %Y').strftime('%Y-%m-%d')
def voltoint(df):
    repl_dict = {'[kK]': '*1e3', '[mM]': '*1e6', '[bB]': '*1e9','-': '-1'}
    df['Vol'] = df['Vol.'].replace(repl_dict, regex=True).map(pd.eval)
def changetoint(df):
    newlist = []
    for i in df['Change %']:
        newlist.append(float(i.replace("%", "")))

    df['Change %'] = newlist
    
    
bitcoin = pd.read_csv(r'bitcoin_historical.csv')
binance = pd.read_csv(r'binance_historical.csv')
cardano = pd.read_csv(r'cardano_historical.csv')
compound = pd.read_csv(r'compound_historical.csv')
dogecoin = pd.read_csv(r'dogecoin_historical.csv')
safemoon = pd.read_csv(r'safemoon_historical.csv')
solana = pd.read_csv(r'solana_historical.csv')
tether =pd.read_csv(r'tether_historical.csv')


changetoint(bitcoin)
changetoint(binance)
changetoint(cardano)
changetoint(compound)
changetoint(dogecoin)
changetoint(safemoon)
changetoint(solana)
changetoint(tether)

voltoint(bitcoin)
voltoint(binance)
voltoint(cardano)
voltoint(compound)
voltoint(dogecoin)
voltoint(safemoon)
voltoint(solana)
voltoint(tether)

coins = [btc, binance, cardano, compound, dogecoin, safemoon, tether]
b=0

for coin in coins:
    Date = []
    coin['Day'] = pd.DatetimeIndex(coin['Date']).day
    coin['Week'] = pd.DatetimeIndex(coin['Date']).week
    coin['Month'] = pd.DatetimeIndex(coin['Date']).month
    coin['Year'] = pd.DatetimeIndex(coin['Date']).year
    for i in coin['Date']:
        Date.append(datechanger((i)))
    coin['Date']=Date
    
solana['Day'] = pd.DatetimeIndex(solana['Date']).day
solana['Week'] = pd.DatetimeIndex(solana['Date']).week
solana['Month'] = pd.DatetimeIndex(solana['Date']).month
solana['Year'] = pd.DatetimeIndex(solana['Date']).year

price = []
inbetween = []
for i in btc['Price']:
    for number in i:
        if number == ',':
            continue
        else:
            inbetween.append(number)
    price.append(float(listtostring(inbetween)))
    inbetween = []
btc['Price'] = price

bitcoin = bitcoin.sort_values(by='Date', ascending=True)
binance = binance.sort_values(by='Date', ascending=True)
cardano = cardano.sort_values(by='Date', ascending=True)
compound = compound.sort_values(by='Date', ascending=True)
dogecoin = dogecoin.sort_values(by='Date', ascending=True)
safemoon = safemoon.sort_values(by='Date', ascending=True)
solana = solana.sort_values(by='Date', ascending=True)
tether = tether.sort_values(by='Date', ascending=True)

def datechanger2(d):
    return datetime.strptime(d, '%d/%m/%Y').strftime('%Y-%m-%d')
def make_366df(data):
    dataframe = data.groupby('Date')['polarity'].mean()
    dataframe = dataframe.to_frame()
    dataframe['date'] = list(set(data['Date']))
    dataframe = dataframe.sort_values(by='date', ascending=True)
    comments = data.groupby('Date')['Total No. of Comments'].sum()
    dataframe['Comments'] = list(comments)
    polarity_change = data.groupby('Date')['polarity%'].mean()
    dataframe['Polarity_change'] = list(polarity_change)
    return dataframe

df = pd.read_csv(r'dataframe_mv.csv').drop(['Unnamed: 0','time stamp', 'date_nr'], axis=1)

Date = []
for i in df['date']:
    Date.append(datechanger2((i)))
df['Date']=Date
df['polarity%'] = df['polarity%'].astype(float)

df_binance = make_366df(df[:15177])
df_bitcoin = make_366df(df[15177:43706])
df_cardano = make_366df(df[43706:45624])
df_compound = make_366df(df[45624:46061])
df_dogecoin = make_366df(df[46061:54239])
df_safemoon = make_366df(df[56069:59331])
df_solana = make_366df(df[59331:61443])
df_tether = make_366df(df[61443:63430])

groupedpolarity = pd.read_csv(r'polarizeddates.csv', sep = ';')
groupedpolarity = groupedpolarity.drop(columns=['Unnamed: 0'])

p_binance = groupedpolarity[0:366]
p_bitcoin = groupedpolarity[366:732]
p_cardano = groupedpolarity[732:1098]
p_compound = groupedpolarity[1098:1464]
p_dogecoin = groupedpolarity[1464:1830]
p_safemoon = groupedpolarity[2196:2562]
p_solana = groupedpolarity[2562:2928]
p_tether = groupedpolarity[2928:3294]

newlist = []
for i in p_compound['Polarity']:
    newlist.append(i)
compound['Polarity'] = newlist

binance['Polarity']= list(p_binance['Polarity'])
bitcoin['Polarity'] = list(p_bitcoin['Polarity'])
cardano['Polarity'] = list(p_cardano['Polarity'])
dogecoin['Polarity']= list(p_dogecoin['Polarity'])
safemoon['Polarity']= list(p_safemoon['Polarity'][269:])
solana['Polarity']= list(p_solana['Polarity'])
tether['Polarity']= list(p_tether['Polarity'])
compound['Polarity'] = p_compound['polarity']

def combinedf(data, data366):
    newlist = []
    newlistpolarityperc = []
    b = 0
    for i in range(len(data)):
        if data['Date'].iloc[i] == data366['date'].iloc[b]:
            newlist.append(data366['Comments'][b])
            newlistpolarityperc.append(data366['Polarity_change'][b])
            b+=1
        else:
            newlist.append('NaN')
            newlistpolarityperc.append('NaN')
    return newlist, newlistpolarityperc

newlist = [0]
newlistpolarityperc = [0]
b = 0
for i in range(len(compound)-1):
    if compound['Date'].iloc[i] == df_compound['date'].iloc[b]:
        newlist.append(df_compound['Comments'][b])
        newlistpolarityperc.append(df_compound['Polarity_change'][b])
        b+=1
    else:
        newlist.append('NaN')
        newlistpolarityperc.append('NaN')

compound['Comments'] = newlist
compound['Polarity_change'] = newlistpolarityperc

newlist = []
newlistpolarityperc = []
b = 98
for i in range(len(safemoon)):
    if safemoon['Date'].iloc[i] == df_safemoon['date'].iloc[b]:
        newlist.append(df_safemoon['Comments'][b])
        newlistpolarityperc.append(df_safemoon['Polarity_change'][b])
        b+=1
    else:
        newlist.append('NaN')
        newlistpolarityperc.append('NaN')

safemoon['Comments'] = newlist
safemoon['Polarity_change'] = newlistpolarityperc

binance['Comments'] = combinedf(binance, df_binance)[0]
btc['Comments'] = combinedf(btc, df_bitcoin)[0]
cardano['Comments'] = combinedf(cardano, df_cardano)[0]
compound['Comments'] = combinedf(compound, df_compound)[0]
dogecoin['Comments'] = combinedf(dogecoin, df_dogecoin)[0]
safemoon['Comments'] = combinedf(safemoon, df_safemoon)[0]
solana['Comments'] = combinedf(solana, df_solana)[0]
tether['Comments'] = combinedf(tether, df_tether)[0]

binance['Polarity_change'] = combinedf(binance, df_binance)[1]
btc['Polarity_change'] = combinedf(btc, df_bitcoin)[1]
cardano['Polarity_change'] = combinedf(cardano, df_cardano)[1]
compound['Polarity_change'] = combinedf(compound, df_compound)[1]
dogecoin['Polarity_change'] = combinedf(dogecoin, df_dogecoin)[1]
safemoon['Polarity_change'] = combinedf(safemoon, df_safemoon)[1]
solana['Polarity_change'] = combinedf(solana, df_solana)[1]
tether['Polarity_change'] = combinedf(tether, df_tether)[1]

def prepforlr(df):
    df = df.dropna()
    return df

bitcoin = prepforlr(bitcoin)
binance = prepforlr(binance)
cardano = prepforlr(cardano)
compound = prepforlr(compound)
dogecoin = prepforlr(dogecoin)
safemoon = prepforlr(safemoon)
solana = prepforlr(solana)
tether = prepforlr(tether)

coins = [binance, bitcoin, cardano, compound, dogecoin, safemoon, solana, tether]
coinname = ['Binance', 'Bitcoin', 'Cardano', 'Compound','Dogecoin', 'Safemoon', 'Solana', 'Tether']
b = 0
for coin in coins:
    print(coinname[b])

    fig, ax = plt.subplots(figsize=(40,8))
    ax.set_title('Polarity and Market Value', size = 35)
    ax.set_xlabel('Date' ,size = 35)
    ax.plot(coin['Date'], coin['Polarity'], color='darkred')
    ax.set_ylabel('Polarity', size = 35)
    ax.set_ylim([-1, 1])
    ax.legend(['Polarity'])
    ax.set_xticks(coin['Date'])
    ax.set_xticklabels(coin['Date'], rotation=90)
    ax.yaxis.grid(color='lightgray', linestyle='dashed')
    plt.tight_layout()
    plt.savefig("polarityvalue{b}.png".format(b=b))
    plt.show()
    b+=1

fulldataset = pd.concat((binance_lr,btc_lr,cardano_lr,compound_lr,dogecoin_lr,polygon_lr,safemoon_lr,solana_lr,tether_lr))
fulldataset.to_csv('fulldataset.csv')


# # Pearson Correlations

# In[ ]:


from scipy import stats
coins = [binance, bitcoin, cardano, compound, dogecoin, safemoon, solana, tether]
coinname = ['binance', 'bitcoin', 'cardano', 'compound','dogecoin', 'safemoon', 'solana', 'tether']
b = 0
for coin in coins:
    print(coinname[b])
    print(stats.pearsonr(coin['Polarity'], coin['Polarity_change']))
    print(stats.pearsonr(coin['Comments'], coin['Change %']))

    b+=1

weak = pd.concat((compound, dogecoin, safemoon, tether))
strong = pd.concat((bitcoin, binance, cardano, solana))

coins = [weak, strong]
coinname = ['weak', 'strong']
b = 0
for coin in coins:
    print(coinname[b])
    print(stats.pearsonr(coin['Polarity'], coin['Change %']))
    print(stats.pearsonr(coin['Comments'], coin['Change %']))

    b+=1    

    

