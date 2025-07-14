#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install xgboost')
get_ipython().system('pip install tensorflow')


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')

import re  
from nltk import word_tokenize
from nltk.tokenize import WordPunctTokenizer
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer

from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
# import string
import warnings

from wordcloud import WordCloud

from sklearn.preprocessing import LabelEncoder,LabelBinarizer

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, BaggingClassifier
from xgboost import XGBClassifier

from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report,precision_score, recall_score,roc_curve, roc_auc_score, auc

import tensorflow as tf
from tensorflow import keras
from sklearn.utils import class_weight
from sklearn.preprocessing import label_binarize
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D,Dropout,GRU
from keras.models import Sequential
#from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import RandomizedSearchCV, KFold
from sklearn.preprocessing import MinMaxScaler


import imblearn
import warnings
warnings.filterwarnings("ignore")


# In[2]:


train=pd.read_csv('data/train_data.csv')
train.head()


# In[3]:


test_val=pd.read_csv('data/test_data_hidden.csv')
test_val.head()


# In[4]:


test= pd.read_csv("data/test_data.csv")
test.head()


# ### Exploratory Data Analysis

# In[5]:


train.duplicated().sum(), test.duplicated().sum(), test_val.duplicated().sum()


# Train dataset contains 58 duplicate records and train dataset contains 3 duplicate records.

# In[6]:


train.drop_duplicates(inplace=True)


# In[7]:


train.duplicated().sum()


# In[8]:


train.shape


# In[9]:


train.info()


# In[10]:


test_val.info()


# In[ ]:





# Train dataset contains 10 missing values in 'reviews.title' column and test dataset contains 3 missing values in 'reviews.title' column.

# In[11]:


#pd.set_option('display.max_colwidth',200)


# Reviews containing Positive Sentiments

# In[12]:


train[train.sentiment=='Positive'][['reviews.text','reviews.title']].head(10)


# In[13]:


train[train.sentiment=='Neutral'][['reviews.text','reviews.title']].head(10)


# In[14]:


train[train.sentiment=='Negative'][['reviews.text','reviews.title']].head(10)


# In[15]:


train.sentiment.value_counts()


# ### Class Imbalance Problem
# In the train dataset, we have 3,749 (~95.1%) sentiments labeled as positive, and 1,58 (~4%) sentiments labeled as Neutral and 93(~2.35%) sentiments as Negative. So, it is an imbalanced classification problem.

# In[16]:


pd.DataFrame(train.name.value_counts())


# In[17]:


#name = pd.DataFrame(train.name.str.split(',').tolist()).stack().unique()
#name = pd.DataFrame(name,columns=['name'])
#name


# In[18]:


train.brand.value_counts() , test_val.brand.value_counts()


# In[19]:


train.primaryCategories.value_counts()


# In[20]:


test_val.primaryCategories.value_counts()


# In[21]:


pd.DataFrame(train.categories.value_counts())


# In[22]:


#categories = pd.DataFrame(train.categories.str.split(',').tolist()).stack().unique()
#categories = pd.DataFrame(categories,columns=['Categories'])
#categories


# In[23]:


train.dtypes


# ### Data Cleaning

# In[24]:


# Removing brand column
del train['brand']
del test_val['brand']
del test['brand']

# New columns - Day, Month, Year from date column 
train['reviews.date'] = train['reviews.date'].str.split('T').str[0]
test_val['reviews.date'] = test_val['reviews.date'].str.split('T').str[0]
test['reviews.date'] = test['reviews.date'].str.split('T').str[0]

train['reviews_day'] = pd.to_datetime(train['reviews.date'], format='%Y-%m-%d').dt.day
train['reviews_month'] = pd.to_datetime(train['reviews.date'], format='%Y-%m-%d').dt.month
train['reviews_year'] = pd.to_datetime(train['reviews.date'], format='%Y-%m-%d').dt.year

test_val['reviews_day'] = pd.to_datetime(test_val['reviews.date'], format='%Y-%m-%d').dt.day
test_val['reviews_month'] = pd.to_datetime(test_val['reviews.date'], format='%Y-%m-%d').dt.month
test_val['reviews_year'] = pd.to_datetime(test_val['reviews.date'], format='%Y-%m-%d').dt.year

test['reviews_day'] = pd.to_datetime(test['reviews.date'], format='%Y-%m-%d').dt.day
test['reviews_month'] = pd.to_datetime(test['reviews.date'], format='%Y-%m-%d').dt.month
test['reviews_year'] = pd.to_datetime(test['reviews.date'], format='%Y-%m-%d').dt.year

del train['reviews.date']
del test['reviews.date']
del test_val['reviews.date']

train.head()


# In[25]:


# Lebel Encoding for item names, categories, primary categories
name = list(set(list(train['name'])+list(test_val['name'])))
categories = list( set( list( train['categories']) + list(test_val['categories'])))
primaryCategories = list(train['primaryCategories'].unique())

le_name = LabelEncoder()
le_cat = LabelEncoder()
le_pri = LabelEncoder()
le_name.fit(name)
le_cat.fit(categories)
le_pri.fit(primaryCategories)

train['name'] = le_name.transform(train.name)
train['categories'] = le_cat.transform(train.categories)
train['primaryCategories'] = le_pri.transform(train.primaryCategories)
test_val['name'] = le_name.transform(test_val.name)
test_val['categories'] = le_cat.transform(test_val.categories)
test_val['primaryCategories'] = le_pri.transform(test_val.primaryCategories)
test['name'] = le_name.transform(test.name)
test['categories'] = le_cat.transform(test.categories)
test['primaryCategories'] = le_pri.transform(test.primaryCategories)


# In[26]:


# Missing Values 
train['reviews.title'].fillna(value=' ',inplace=True)
test_val['reviews.title'].fillna(value=' ',inplace=True)
test['reviews.title'].fillna(value=' ',inplace=True)


# In[27]:


# Text data cleaning : reviews.text, reviews.title

tok = WordPunctTokenizer()
ps = PorterStemmer()
wnl = WordNetLemmatizer()

negations_dic = {"isn't":"is not", "aren't":"are not", "wasn't":"was not", "weren't":"were not",
                "haven't":"have not","hasn't":"has not","hadn't":"had not","won't":"will not",
                "wouldn't":"would not", "don't":"do not", "doesn't":"does not","didn't":"did not",
                "can't":"can not","couldn't":"could not","shouldn't":"should not","mightn't":"might not",
                "mustn't":"must not"}

neg_pattern = re.compile(r'\b(' + '|'.join(negations_dic.keys()) + r')\b')

def data_cleaner(text):
    text = text.replace(r"Äú",'')
    text = text.replace(r'Äù','')
    text = text.replace(r'‚Äô','\'')
    text = text.lower()
    text = text.replace(r'‚Äô','\'')
    text = neg_pattern.sub(lambda x: negations_dic[x.group()], text)
    text = re.sub("[^a-zA-Z0-9\"]", " ", text)
    word_tok=[x for x in tok.tokenize(text) if len(x) > 3]
#     word_stem = [ps.stem(i) for i in word_tok]
#     return (" ".join(word_stem).strip())  
    word_lem = [wnl.lemmatize(i) for i in word_tok]
    return (" ".join(word_lem).strip()) 

for i in (train,test_val,test):
    i['reviews.text']=i['reviews.text'].apply(data_cleaner)
    i['reviews.title']=i['reviews.title'].apply(data_cleaner) 


# In[28]:


test[['reviews.text','reviews.title']].head(10)


# ### Visualization

# In[29]:


train_len=train["reviews.text"].str.len()
test_len=test["reviews.text"].str.len()
plt.hist(train_len,bins=20,label="train reviews")
plt.hist(test_len,bins=20,label="test reviews")
plt.legend()
plt.xlim(0,2000)
plt.xlabel('Characters')
plt.ylabel('Reviews Count')
plt.show()


# In[30]:


all_text = ' '.join([text for text in train['reviews.text']])
pos_text = ' '.join([text for text in train['reviews.text'][train['sentiment']=='Positive']])
neg_text = ' '.join([text for text in train['reviews.text'][train['sentiment']=='Negative']])
neu_text = ' '.join([text for text in train['reviews.text'][train['sentiment']=='Neutral']])


# In[31]:


wordcloud = WordCloud(width=1600, height=800, random_state=21, max_font_size=180).generate(pos_text)
plt.figure(figsize=(12,10))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title(' POSITIVE REVIEWS')
plt.show()


# In[32]:


wordcloud = WordCloud(height=800, width=1600, random_state=21,max_font_size=180).generate(neg_text)
plt.figure(figsize=(12,10))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title(' NEGATIVE REVIEWS')
plt.show()


# In[33]:


wordcloud = WordCloud(height=800, width=1600, random_state=21,max_font_size=180).generate(neu_text)
plt.figure(figsize=(12,10))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('NEUTRAL REVIEWS')
plt.show()


# In[34]:


le_senti = LabelEncoder()
train['sentiment'] = le_senti.fit_transform(train['sentiment'])
test_val['sentiment'] = le_senti.fit_transform(test_val['sentiment'])


# In[35]:


train.head()


# ### TFIDF Vectorizer

# In[36]:


tvec1 = TfidfVectorizer()
tvec2 = TfidfVectorizer()
tvec3 = TfidfVectorizer()


# In[37]:


# Preparing Features X(text,title) and Label y(sentiment)

train1 = train.reset_index()
combi1=pd.concat([train1,test_val],axis=0,join='outer')

tvec1.fit(combi1['reviews.text'])
tvec_text1 = pd.DataFrame(tvec1.transform(train1['reviews.text']).toarray())
tvec_text2 = pd.DataFrame(tvec1.transform(test_val['reviews.text']).toarray())

tvec2.fit(combi1['reviews.title'])
tvec_title1 = pd.DataFrame(tvec2.transform(train1['reviews.title']).toarray())
tvec_title2 = pd.DataFrame(tvec2.transform(test_val['reviews.title']).toarray())

Train1 = pd.concat([train1.drop(['reviews.text','reviews.title','sentiment','index'],axis=1),tvec_text1, tvec_title1],axis=1)
Test_Val1 = pd.concat([test_val.drop(['reviews.text','reviews.title','sentiment'],axis=1),tvec_text2, tvec_title2],axis=1)

x_train1=Train1.values
y_train1=train['sentiment'].values

x_val1=Test_Val1.values
y_val1 = test_val['sentiment'].values


# In[38]:


from nltk.tokenize import RegexpTokenizer
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction import text

punc = ['.', ',', '"', "'", '?', '!', ':', ';', '(', ')', '[', ']', '{', '}',"%"]
stop_words = text.ENGLISH_STOP_WORDS.union(punc)

stemmer = SnowballStemmer('english')
tokenizer = RegexpTokenizer(r'[a-zA-Z\']+')

def tokenize(text):
    return [stemmer.stem(word) for word in tokenizer.tokenize(text.lower())]

tvec3 = TfidfVectorizer(stop_words = list(stop_words), tokenizer = tokenize, max_features = 1000)

reviews=tvec3.fit_transform(combi1['reviews.text'])
words = tvec3.get_feature_names_out()


# ### Multinomial Naive Bayes

# In[39]:


nb = MultinomialNB()
nb.fit(Train1.values,train1['sentiment'])
y_pred = nb.predict(Test_Val1.values)
y_val = test_val['sentiment']
print(confusion_matrix(y_true=y_val, y_pred=y_pred))
print(classification_report(y_true=y_val, y_pred=y_pred))
print(accuracy_score(y_val, y_pred)*100)


# Everything is classified as Positive because of Imbalance Class

# 
# #### Tackling Class Imbalance Problem:

# In[40]:


train.sentiment.value_counts()


# In[41]:


count_2, count_1, count_0 =train.sentiment.value_counts()
class_2 = train[train.sentiment==2]
class_1 = train[train.sentiment==1]
class_0 = train[train.sentiment==0]
count_2, count_1, count_0


# #### 1. UnderSampling

# In[42]:


class_2_under = class_2.sample(count_1)
train_under= pd.concat([class_2_under,class_1,class_0],axis=0)
print(train_under.shape)
print(train_under.sentiment.value_counts())


# #### 2. OverSampling
# 

# In[43]:


class_0_over = class_0.sample(count_2,replace=True)
class_1_over = class_1.sample(count_2,replace=True)
train_over = pd.concat([class_2,class_0_over,class_1_over],axis=0)
print(train_over.shape)
print(train_over.sentiment.value_counts())


# In[44]:


lr= LogisticRegression(C=30, class_weight='balanced', solver='sag', 
                         multi_class='multinomial', n_jobs=6, random_state=40, 
                         verbose=1, max_iter=1000)


# #### TFIDF Vectorizer for under-sampled data

# In[45]:


train = train_under.reset_index(drop=True) 
#combi = train.append(test_val,ignore_index=True)
combi=pd.concat([train,test_val],axis=0,join='outer')
print(combi.shape)

tvec1.fit(combi['reviews.text'])
tvec_text1 = pd.DataFrame(tvec1.transform(train['reviews.text']).toarray())
tvec_text2 = pd.DataFrame(tvec1.transform(test_val['reviews.text']).toarray())

tvec2.fit(combi['reviews.title'])
tvec_title1 = pd.DataFrame(tvec2.transform(train['reviews.title']).toarray())
tvec_title2 = pd.DataFrame(tvec2.transform(test_val['reviews.title']).toarray())

Train = pd.concat([train.drop(['reviews.text','reviews.title','sentiment'],axis=1),tvec_text1, tvec_title1],axis=1)
Test_Val = pd.concat([test_val.drop(['reviews.text','reviews.title','sentiment'],axis=1),tvec_text2, tvec_title2],axis=1)
x_train=Train.values
y_train=train['sentiment']
x_val=Test_Val.values
y_val = test_val['sentiment']


# ### Logistic Regression for under-sampled data

# In[46]:


lr.fit(x_train,y_train)
y_pred = lr.predict(x_val)
print(confusion_matrix(y_true=y_val, y_pred=y_pred))
print(classification_report(y_true=y_val, y_pred=y_pred))
print('accuracy : ',accuracy_score(y_val, y_pred)*100)


# In[47]:


lb = LabelBinarizer()
lb.fit(y_val)
y_val1 = lb.transform(y_val)
y_pred1 = lb.transform(y_pred)
print(roc_auc_score(y_val1, y_pred1, average='weighted'))
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(3):
    fpr[i], tpr[i], _ = roc_curve(y_val1[:, i], y_pred1[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

lw=2
for i in range(3):
    plt.plot(fpr[i], tpr[i], lw=lw,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic of Logistic Regression of under -sampled data')
plt.legend(loc="lower right")
plt.show()


# #### TFIDF Vectorizer for over-sampled data

# In[48]:


train = train_over.reset_index(drop=True) 

tvec1.fit(train['reviews.text'])
tvec_text1 = pd.DataFrame(tvec1.transform(train['reviews.text']).toarray())
tvec_text2 = pd.DataFrame(tvec1.transform(test_val['reviews.text']).toarray())

tvec2.fit(train['reviews.title'])
tvec_title1 = pd.DataFrame(tvec2.transform(train['reviews.title']).toarray())
tvec_title2 = pd.DataFrame(tvec2.transform(test_val['reviews.title']).toarray())

Train = pd.concat([train.drop(['reviews.text','reviews.title','sentiment'],axis=1),tvec_text1, tvec_title1],axis=1)
Test_Val = pd.concat([test_val.drop(['reviews.text','reviews.title','sentiment'],axis=1),tvec_text2, tvec_title2],axis=1)

Train.to_csv('Train.csv',encoding='utf-8')
Test_Val.to_csv('Test_Val.csv',encoding='utf-8')

x_train=Train.values
y_train=train['sentiment'].values
x_val=Test_Val.values
y_val = test_val['sentiment'].values


# ### Logistic Regression for over-sampled data

# In[49]:


lr.fit(x_train,y_train)
y_pred = lr.predict(x_val)
print(confusion_matrix(y_true=y_val, y_pred=y_pred))
print(classification_report(y_true=y_val, y_pred=y_pred))
print('accuracy : ',accuracy_score(y_val, y_pred)*100)


# Logistic Regression on over-sampled data is perfrorming better than under-sampled data

# In[50]:


lb = LabelBinarizer()
lb.fit(y_val)
y_val1 = lb.transform(y_val)
y_pred1 = lb.transform(y_pred)
print(roc_auc_score(y_val1, y_pred1, average='weighted'))
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(3):
    fpr[i], tpr[i], _ = roc_curve(y_val1[:, i], y_pred1[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

lw=2
for i in range(3):
    plt.plot(fpr[i], tpr[i], lw=lw,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title(' Receiver operating characteristic for Logistic Regression of over-sampled data')
plt.legend(loc="lower right")
plt.show()


# ### Multinomial Naive Bayes

# In[51]:


nb = MultinomialNB()
nb.fit(x_train,y_train)
y_pred = nb.predict(x_val)
print(confusion_matrix(y_true=y_val, y_pred=y_pred))
print(classification_report(y_true=y_val, y_pred=y_pred))
print(accuracy_score(y_val, y_pred)*100)
print(nb.score(x_train,y_train))
print(nb.score(x_val,y_val))


# In[52]:


lb = LabelBinarizer()
lb.fit(y_val)
y_val1 = lb.transform(y_val)
y_pred1 = lb.transform(y_pred)
print(roc_auc_score(y_val1, y_pred1, average='weighted'))
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(3):
    fpr[i], tpr[i], _ = roc_curve(y_val1[:, i], y_pred1[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

lw=2
for i in range(3):
    plt.plot(fpr[i], tpr[i], lw=lw,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic of Multinomial Naive Bayes Classifier')
plt.legend(loc="lower right")
plt.show()


# ### Random Forest Classifier

# In[53]:


rf= RandomForestClassifier(n_estimators=400,random_state=10).fit(x_train,y_train)
y_pred=rf.predict(x_val)
print(confusion_matrix(y_true=y_val, y_pred=y_pred))
print(classification_report(y_true=y_val, y_pred=y_pred))
print('accuracy : ',accuracy_score(y_val, y_pred)*100)
print(rf.score(x_train,y_train))
print(rf.score(x_val,y_val))


# In[54]:


lb = LabelBinarizer()
lb.fit(y_val)
y_val1 = lb.transform(y_val)
y_pred1 = lb.transform(y_pred)
print(roc_auc_score(y_val1, y_pred1, average='weighted'))
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(3):
    fpr[i], tpr[i], _ = roc_curve(y_val1[:, i], y_pred1[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

lw=2
for i in range(3):
    plt.plot(fpr[i], tpr[i], lw=lw,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic of Random Forest Classifier')
plt.legend(loc="lower right")
plt.show()


# ### XGBClassifier

# In[55]:


xgb= XGBClassifier(n_estimators=1000,max_depth=6).fit(x_train,y_train)
y_pred=xgb.predict(x_val)
print(confusion_matrix(y_true=y_val, y_pred=y_pred))
print(classification_report(y_true=y_val, y_pred=y_pred))
print("accuracy : ",accuracy_score(y_val, y_pred)*100)


# In[56]:


lb = LabelBinarizer()
lb.fit(y_val)
y_val1 = lb.transform(y_val)
y_pred1 = lb.transform(y_pred)
print(roc_auc_score(y_val1, y_pred1, average='weighted'))
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(3):
    fpr[i], tpr[i], _ = roc_curve(y_val1[:, i], y_pred1[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

lw=2
for i in range(3):
    plt.plot(fpr[i], tpr[i], lw=lw,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic of XGBClassifier')
plt.legend(loc="lower right")
plt.show()


# We can see that XGBoost is performing better in predicting all the classes.

# ### Multi-class SVM

# In[57]:


svc = SVC(kernel='linear', class_weight='balanced', C=1.0, random_state=0).fit(x_train, y_train) 
y_pred=svc.predict(x_val)
print(confusion_matrix(y_true=y_val, y_pred=y_pred))
print(classification_report(y_true=y_val, y_pred=y_pred))
print("accuracy : ",accuracy_score(y_val, y_pred)*100)


# In[58]:


lb = LabelBinarizer()
lb.fit(y_val)
y_val1 = lb.transform(y_val)
y_pred1 = lb.transform(y_pred)
print(roc_auc_score(y_val1, y_pred1, average='weighted'))
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(3):
    fpr[i], tpr[i], _ = roc_curve(y_val1[:, i], y_pred1[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

lw=2
for i in range(3):
    plt.plot(fpr[i], tpr[i], lw=lw,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic of multiclass SVM Classifier')
plt.legend(loc="lower right")
plt.show()


# Naive Bayes
# 

# In[59]:


y_train2 = label_binarize(y_train1, classes=[0, 1, 2])


# In[60]:


#The model with sequential API
classifier = Sequential()
classifier.add(Dense(units=100,kernel_initializer='he_uniform',activation='relu',input_dim=x_train1.shape[1]))
classifier.add(Dense(units=80,kernel_initializer='he_uniform',activation='relu'))
classifier.add(Dense(units=80,kernel_initializer='he_uniform',activation='relu'))
classifier.add(Dense(units=3,kernel_initializer='normal',activation='softmax'))
#Compile and Run
classifier.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
classifier.fit(x_train1,y_train2,batch_size=256,epochs=100,verbose=0)
#Evaluate
y_pred = classifier.predict(x_val1, batch_size=256)

# Confusion matrix needs both labels & predictions as single-digits, not as one-hot encoded vectors
# predictions are the probabilities, and when np.argmax(..) is applied, it gives the predicted label.
y_pred_bool = np.argmax(y_pred, axis=1)
# label
y_test = np.argmax(y_val1, axis=1) # one hot encoding to a single number for test set

print(confusion_matrix(y_test, y_pred_bool))
print(classification_report(y_test, y_pred_bool))


# In[61]:


class_weights = class_weight.compute_class_weight(class_weight='balanced',
                                                  classes=np.unique(y_train1),
                                                  y=y_train1)
class_weight_dict = dict(enumerate(class_weights))
class_weight_dict


# In[62]:


# Using Class-Weights
classifier = Sequential()
classifier.add(Dense(units=50,activation='relu',input_dim=x_train1.shape[1]))
classifier.add(Dense(units=40,activation='relu'))
classifier.add(Dense(units=3,kernel_initializer='normal',activation='softmax'))
classifier.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
classifier.fit(x_train1,y_train2,batch_size=256,epochs=100,class_weight=class_weight_dict,verbose=0)
y_pred = classifier.predict(x_val1, batch_size=256)
y_pred_bool = np.argmax(y_pred, axis=1)
y_test = np.argmax(y_val1, axis=1)
print(confusion_matrix(y_test, y_pred_bool))
print(classification_report(y_test, y_pred_bool))


# Using class-weights does not improve the performance

# In[63]:


#using dropouts
classifier = Sequential()
classifier.add(Dense(units=50,activation='relu',input_dim=x_train1.shape[1]))
classifier.add(Dropout(0.2))
classifier.add(Dense(units=40,activation='relu'))
classifier.add(Dropout(0.2))
classifier.add(Dense(units=40,activation='relu'))
classifier.add(Dense(units=3,kernel_initializer='normal',activation='softmax'))
classifier.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
classifier.fit(x_train1,y_train2,batch_size=256,epochs=100,class_weight=class_weight_dict,verbose=0)
y_pred = classifier.predict(x_val1, batch_size=256)
y_pred_bool = np.argmax(y_pred, axis=1)
y_test = np.argmax(y_val1, axis=1)
print(confusion_matrix(y_test, y_pred_bool))
print(classification_report(y_test, y_pred_bool))


# Using drop out chances of predicting second class increases

# In[64]:


y_train3 = label_binarize(y_train, classes=[0, 1, 2])


# In[65]:


#for over-sampled data
classifier = Sequential()
classifier.add(Dense(units=50,activation='relu',input_dim=x_train.shape[1]))
classifier.add(Dense(units=40,activation='relu'))
classifier.add(Dense(units=150,activation='relu'))
classifier.add(Dense(units=3,kernel_initializer='normal',activation='softmax'))
classifier.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
classifier.fit(x_train,y_train3,batch_size=256,epochs=10,verbose=0)
y_pred = classifier.predict(x_val, batch_size=256)
y_pred_bool = np.argmax(y_pred, axis=1)
y_test = np.argmax(y_val1, axis=1)
print(confusion_matrix(y_test, y_pred_bool))
print(classification_report(y_test, y_pred_bool))


# Using Over-sampled data for neural network does not improve the performance
# 
# ### ensemble technique using Voting Classifier: XGboost + oversampled_multinomial_NB

# In[66]:


from sklearn.ensemble import VotingClassifier
model1 = MultinomialNB()
model2 =  XGBClassifier(n_estimators=1000,max_depth=6)
model = VotingClassifier(estimators=[('lr', model1), ('dt', model2)], voting='hard')
model.fit(x_train,y_train)
y_pred = model.predict(x_val)
print(confusion_matrix(y_true=y_val, y_pred=y_pred))
print(classification_report(y_true=y_val, y_pred=y_pred))
print("accuracy : ",accuracy_score(y_val, y_pred)*100)


# We can see that the above model performs almost same as oversampled multinomial model but it increases the chances of prediction of minority classes.
# 
# ### Sentiment Score

# In[67]:


get_ipython().system('pip install textblob')

from textblob import TextBlob
import nltk
nltk.download('punkt')

def senti(x):
    return TextBlob(x).sentiment

def polarity(x):
    return TextBlob(x).sentiment.polarity + 1

train['senti_score'] = train['reviews.text'].apply(senti)
test_val['senti_score'] = test_val['reviews.text'].apply(senti)

train['polarity'] = train['reviews.text'].apply(polarity)
test_val['polarity'] = test_val['reviews.text'].apply(polarity)

train['senti_score'].head()


# In[68]:


Train = pd.concat([train.drop(['reviews.text','reviews.title','sentiment','senti_score'],axis=1),tvec_text1, tvec_title1],axis=1)
Test_Val = pd.concat([test_val.drop(['reviews.text','reviews.title','sentiment','senti_score'],axis=1),tvec_text2, tvec_title2],axis=1)
x_train=Train.values
y_train=train['sentiment']
x_val=Test_Val.values
y_val = test_val['sentiment']


# In[69]:


nb = MultinomialNB()
nb.fit(x_train,y_train)
y_pred = nb.predict(x_val)
print(confusion_matrix(y_true=y_val, y_pred=y_pred))
print(classification_report(y_true=y_val, y_pred=y_pred))
print(accuracy_score(y_val, y_pred)*100)
print(nb.score(x_train,y_train))
print(nb.score(x_val,y_val))


# Sentiment Score does not have much affect on the performance
# 
# 
# #### LSTM

# In[70]:


from sklearn.preprocessing import label_binarize
from tensorflow.keras.optimizers import Adam


# One-hot encode target labels
y_train2 = label_binarize(y_train1, classes=[0, 1, 2])
y_val2 = label_binarize(y_val1, classes=[0, 1, 2])

# Parameters — reduced
epochs = 3
emb_dim = 64              # reduced embedding
batch_size = 64           # reduced batch size

# Build model
model = Sequential()
model.add(Embedding(5000, emb_dim, input_length=x_train1.shape[1]))
model.add(SpatialDropout1D(0.3))  # lower dropout
model.add(LSTM(32, dropout=0.3, recurrent_dropout=0.3))  # smaller LSTM
model.add(Dense(3, activation='softmax'))

model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Train model
model.fit(x_train1, y_train2, epochs=epochs, batch_size=batch_size, validation_data=(x_val1, y_val2))

# Predict
y_pred = model.predict(x_val1, batch_size=32)
y_pred_bool = np.argmax(y_pred, axis=1)
y_test = np.argmax(y_val2, axis=1)

from sklearn.metrics import confusion_matrix, classification_report
print(confusion_matrix(y_test, y_pred_bool))
print(classification_report(y_test, y_pred_bool))


# In[71]:


from sklearn.preprocessing import label_binarize
from sklearn.utils import class_weight
from sklearn.metrics import confusion_matrix, classification_report

# One-hot encode labels
y_train2 = label_binarize(y_train1, classes=[0, 1, 2])
y_val2 = label_binarize(y_val1, classes=[0, 1, 2])

# Compute class weights
class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train1),
    y=y_train1
)
class_weights_dict = dict(enumerate(class_weights))

# Key correction: vocab size
vocab_size = 5000           # or whatever was used in your tokenizer
input_len = x_train1.shape[1]

# Safer model
emb_dim = 64
epochs = 4
batch_size = 64             # reduce batch size for stability

model = Sequential()
model.add(Embedding(input_dim=vocab_size, output_dim=emb_dim, input_length=input_len))
model.add(SpatialDropout1D(0.3))  # lowered for stability
model.add(LSTM(32, dropout=0.3, recurrent_dropout=0.3))
model.add(Dense(3, activation='softmax'))

model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Train model with class weights
model.fit(x_train1, y_train2, epochs=epochs, batch_size=batch_size,
          validation_data=(x_val1, y_val2),
          class_weight=class_weights_dict)

# Predict and evaluate
y_pred = model.predict(x_val1, batch_size=32)
y_pred_bool = np.argmax(y_pred, axis=1)
y_test = np.argmax(y_val2, axis=1)

print(confusion_matrix(y_test, y_pred_bool))
print(classification_report(y_test, y_pred_bool))


# In[72]:


from sklearn.preprocessing import label_binarize
from sklearn.metrics import confusion_matrix, classification_report


# One-hot encode labels
y_train2 = label_binarize(y_train, classes=[0, 1, 2])
y_val2 = label_binarize(y_val1, classes=[0, 1, 2])

# Define parameters
vocab_size = 5000        # use the vocab size from your tokenizer
input_len = x_train.shape[1]

emb_dim = 64             # reduced to save memory
epochs = 3
batch_size = 64          # much safer

# Build model
model = Sequential()
model.add(Embedding(input_dim=vocab_size, output_dim=emb_dim, input_length=input_len))
model.add(SpatialDropout1D(0.3))
model.add(LSTM(32, dropout=0.3, recurrent_dropout=0.3))
model.add(Dense(3, activation='softmax'))

model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Train
model.fit(x_train, y_train2, epochs=epochs, batch_size=batch_size, validation_data=(x_val, y_val2))

# Predict and evaluate
y_pred = model.predict(x_val, batch_size=32)
y_pred_bool = np.argmax(y_pred, axis=1)
y_test = np.argmax(y_val2, axis=1)

print(confusion_matrix(y_test, y_pred_bool))
print(classification_report(y_test, y_pred_bool))


# #### GRU

# In[73]:


from sklearn.preprocessing import label_binarize
from sklearn.metrics import confusion_matrix, classification_report


# One-hot encode labels
y_train2 = label_binarize(y_train1, classes=[0, 1, 2])
y_val2 = label_binarize(y_val1, classes=[0, 1, 2])

# Define correct vocab size and sequence length
vocab_size = 5000  # should match your tokenizer's vocab size
sequence_length = x_train1.shape[1]

# Build model
emb_dim = 64              # safer embedding dim
epochs = 3
batch_size = 64           # reduce batch size to prevent RAM issues

model = Sequential()
model.add(Embedding(input_dim=vocab_size, output_dim=emb_dim, input_length=sequence_length))
model.add(GRU(64, dropout=0.3, recurrent_dropout=0.3))
model.add(Dense(3, activation='softmax'))

model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Train model
model.fit(x_train1, y_train2, epochs=epochs, batch_size=batch_size, validation_data=(x_val1, y_val2))

# Evaluate
y_pred = model.predict(x_val1, batch_size=32)
y_pred_bool = np.argmax(y_pred, axis=1)
y_test = np.argmax(y_val2, axis=1)

print(confusion_matrix(y_test, y_pred_bool))
print(classification_report(y_test, y_pred_bool))


# We can see from above that LSTM and GPU models are not efficient in predicting minor classes. ANN is performing quite good in solving class imbalance problem but it cannot beat traditional ML agorithms.
# 
# ### Clustering of Reviews

# In[74]:


print(words[250:300])


# In[75]:


from sklearn.cluster import KMeans
wcss = []
for i in range(1,15):
    kmeans = KMeans(n_clusters=i,init='k-means++',max_iter=300,n_init=10,random_state=0)
    kmeans.fit(reviews)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,15),wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()


# As no proper elbow is generated, I will have to select right amount of clusters by trial and error. So, I will showcase the results of different amount of clusters to find out the right amount of clusters.
# 
# ### 11 Clusters

# In[76]:


kmeans = KMeans(n_clusters = 11, n_init = 20) 
kmeans.fit(reviews)
# We look at 6 the clusters generated by k-means.
common_words = kmeans.cluster_centers_.argsort()[:,-1:-26:-1]
for num, centroid in enumerate(common_words):
    print(str(num) + ' : ' + ', '.join(words[word] for word in centroid))


# ### 13 Clusters

# In[77]:


kmeans = KMeans(n_clusters = 13, n_init = 20) 
kmeans.fit(reviews)
# We look at 13 the clusters generated by k-means.
common_words = kmeans.cluster_centers_.argsort()[:,-1:-26:-1]
for num, centroid in enumerate(common_words):
    print(str(num) + ' : ' + ', '.join(words[word] for word in centroid))


# ### Topic Modelling

# In[78]:


from sklearn.decomposition import LatentDirichletAllocation as LDA
# Helper function
def print_topics(model, count_vectorizer, n_top_words):
    words = count_vectorizer.get_feature_names_out()
    for topic_idx, topic in enumerate(model.components_):
        print("\nTopic #%d:" % topic_idx)
        print(" ".join([words[i]
                        for i in topic.argsort()[:-n_top_words - 1:-1]]))
# Tweak the two parameters below
number_topics = 10
number_words = 10
# Create and fit the LDA model
lda = LDA(n_components=number_topics, n_jobs=-1)
lda.fit(reviews)
# Print the topics found by the LDA model
print("Topics found via LDA:")
print_topics(lda, tvec3, number_words)


# In[79]:


from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer()
X = tfidf.fit_transform(train['reviews.text'])  # use your dataframe column as corpus


# In[80]:


import pickle

# Save model
with open('sentiment_model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Save vectorizer (assuming its name is tfidf)
with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(tfidf, f)


# In[1]:


# train_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib

# Step 1: Load dataset
df = pd.read_csv("train_data.csv")
df = df.dropna(subset=["reviews.text", "label"])

# Step 2: Add custom negation and sentiment examples
custom_examples = pd.DataFrame({
    "reviews.text": [
        "this product is not bad",
        "not bad at all",
        "not horrible",
        "not good",
        "the product is very good",
        "not the worst",
        "not a bad product",
        "it's not awful",
        "not terrible",
        "it's not great",
        "it's not that bad",
        "this product is bad",
        "very bad product",
        "bad quality",
        "worst product ever",
        "not satisfied",
        "really bad experience",
        "complete waste of money",
        "bad build quality",
        "terrible product",
        "did not like the product",
          # Neutral (1)
        "the product works fine but nothing exceptional",
        "average performance, okay for basic tasks",
        "build quality is average but usable",
        "received the item late but it works",
        "not bad, not great either",
        "performance is acceptable for daily use",
        "okay-ish product, nothing special",
        "battery life is decent but not impressive"
    ],
    "label": [
        2, 2, 2, 0, 2, 2, 2, 2, 2, 1, 2,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,1,1,1,1,1,1,1,1
    ]
})

# Step 2b: Add extra positive reviews
custom_positive = pd.DataFrame({
    'reviews.text': [
        'this product is very good',
        'absolutely fantastic product',
        'really satisfied and very happy',
        'worth buying, highly recommended'
    ],
    'label': [2, 2, 2, 2]  # 2 = Positive
})

# Step 3: Combine original + custom examples
df = pd.concat([df, custom_examples, custom_positive], ignore_index=True)

# Step 4: Features and labels
X = df["reviews.text"]
y = df["label"]

# Step 5: Pipeline with bigram TF-IDF
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=5000)),
    ("clf", LogisticRegression(max_iter=1000))
])

# Step 6: Train the model
pipeline.fit(X, y)

# Step 7: Save model
joblib.dump(pipeline, "model.pkl")
print(" Model trained with bigrams + custom reviews — saved to model.pkl")

