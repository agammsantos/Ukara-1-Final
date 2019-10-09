import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import ComplementNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import classification_report,confusion_matrix,f1_score

data=pd.read_csv('data_train_B.csv')
print(len(data))
print(data.groupby(data['LABEL']).count())

grouped=data.groupby('LABEL')
print(grouped.ngroups)
print(grouped.describe())

data['relist']=data['RESPONSE'].apply(lambda x: x.replace(',',' ').replace('.',' ').replace('/',' ').replace(' - ',' ').replace('?','').replace('!','').replace('"','').replace("'",'').split(' ')) # column that contain list of words for each response

k=0
for i in data['relist']: # loop for removing empty elements created by split
    j=0
    length = len(i)
    while j<length:
        if i[j]=='':
            data['relist'].iloc[k].remove('')
            length-=1  
            continue
        j+=1
    k+=1

data['relen']=data['relist'].apply(len) # column that contain length of words for each response
print('The longest sentence has: '+str(data['relen'].max())+' words')
print(data.iloc[data[data['relen']==data['relen'].max()].index.values,1])
print(data.iloc[data[data['relen']==data['relen'].min()].index.values,1])
print(data.iloc[284]['RESPONSE'])
print('\n')
data['words']=data['relist'].apply(lambda x:' '.join(x)).apply(str.lower) # column that contain pre-processed words for each response for further analysis
groupedfix=data.groupby('LABEL')

freqdata=pd.Series(' '.join(data['words']).lower().split(' ')).value_counts()[:5]
print('Most common words in response data: ')
print(freqdata)
print('\n')

print(data.head())
print(data.tail())

common=[]
for i in groupedfix: # creating stopwords from both labels same common words
    freqdatalabel=pd.Series(' '.join(i[1]['words']).lower().split(' ')).value_counts()[:5]
    common.extend(freqdatalabel.index.values)
    print('Most common words for label '+str(i[0])+':')
    print(freqdatalabel)
    print('\n')
comcount=Counter(common)
comcountfix={key:val for key, val in comcount.items() if val!=1} 
print(comcountfix)

stops=list(comcountfix.keys())

# create stemmer
# factory = StemmerFactory()
# stemmer = factory.create_stemmer()

from sklearn.model_selection import train_test_split
xtr,xts,ytr,yts=train_test_split(data['RESPONSE'].apply(lambda x:x.lower()),data['LABEL'],
    stratify=data['LABEL'],
    test_size=0.1, 
    random_state=420)

# nltkstopwds=stopwords.words('indonesian')
# print(stopwords.words('indonesian'))
# print(len(stopwords.words('indonesian')))

# factory = StopWordRemoverFactory()
# stopwds = factory.get_stop_words()
# print(stopwds)
# print(len(stopwds))

# print(len(set(stopwords.words('indonesian')+stopwds)))
# allstopwords=list(set(stopwords.words('indonesian')+stopwds))

test=pd.read_csv('data_test_B.csv')

complementPipeline = Pipeline([
    ('cv',CountVectorizer(
        # stop_words=stops,
        # ngram_range=(1,5)
        )),
    ('tfidf',TfidfTransformer()),
    ('classifier',ComplementNB())
])

customPipeline = Pipeline([
    ('cv',CountVectorizer(
        # stop_words=stops,
        ngram_range=(1,4)
        )),
    ('tfidf',TfidfTransformer(norm='l2',smooth_idf=True,sublinear_tf=False,use_idf=False)),
    ('classifier',ComplementNB())
])

parameters={    
    # 'cv__max_df': (0.5, 0.75, 1.0),
    'cv__stop_words':(stops, None),
    # 'cv__max_features': (None, 5000, 10000, 50000),
    'cv__ngram_range': ((1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (2, 5), (3, 5), (4, 5), (5, 5)),  # unigrams or bigrams
    'tfidf': (None, TfidfTransformer(use_idf=False, norm='l2')),
    # 'tfidf__use_idf': (True, False),
    # 'tfidf__norm': ('l1', 'l2'),
    }

gridSearch = GridSearchCV(complementPipeline, parameters, cv=5, n_jobs=-1, verbose=1, scoring='f1')
gridSearchSplit = GridSearchCV(complementPipeline, parameters, cv=5, n_jobs=-1, verbose=1, scoring='f1')

# complementPipeline.fit(data['words'],data['LABEL'])
gridSearch.fit(data['words'],data['LABEL'])
gridSearchSplit.fit(xtr,ytr)
customPipeline.fit(xtr,ytr)

# complementPrediksiTrain = complementPipeline.predict(data['words'])
# complementPrediksi = complementPipeline.predict(data['words'])
complementPrediksi = gridSearch.predict(data['words'])
complementPrediksiSplit = gridSearchSplit.predict(xts)
complementPrediksiCus = customPipeline.predict(xts)
# complementPrediksiDev = complementPipeline.predict(test['RESPONSE'].apply(lambda x:x.lower()))
complementPrediksiDev = gridSearch.predict(test['RESPONSE'].apply(lambda x:str(x).lower()))

# Evaluasi data keseluruhan
print('Dengan metode ComplementNB, diperoleh: ')
print(classification_report(data['LABEL'],complementPrediksi))
print('Skor f1:',f1_score(data['LABEL'],complementPrediksi))
print(confusion_matrix(data['LABEL'],complementPrediksi))
print(gridSearch.best_score_)
best_parameters = gridSearch.best_estimator_.get_params()
for param_name in sorted(parameters.keys()):
    print("\t%s: %r" % (param_name, best_parameters[param_name]))

# Evaluasi data split
print('Dengan metode ComplementNB, diperoleh: ')
print(classification_report(yts,complementPrediksiSplit))
print('Skor f1:',f1_score(yts,complementPrediksiSplit))
print(confusion_matrix(yts,complementPrediksiSplit))
print(gridSearchSplit.best_score_)
best_parameters_split = gridSearchSplit.best_estimator_.get_params()
for param_name in sorted(parameters.keys()):
    print("\t%s: %r" % (param_name, best_parameters_split[param_name]))

# Evaluasi parameter GridSearch
print('Dengan metode ComplementNB, diperoleh: ')
print(classification_report(yts,complementPrediksiCus))
print('Skor f1:',f1_score(yts,complementPrediksiCus))
print(confusion_matrix(yts,complementPrediksiCus))

# Prediksi menggunakan data traning keseluruhan / 100%
PREDIKSIDEV=enumerate(complementPrediksiDev)
result=[]
for i,j in PREDIKSIDEV:
    result.append({"RES_ID":test['RES_ID'][i],"LABEL":int(j)})

import json
with open('data1mix.json','r') as json_file:
  data1mix=json.load(json_file)
  data1mix.extend(result)

with open('datapredictmix.json','w') as jsfix:  
  json.dump(data1mix, jsfix)