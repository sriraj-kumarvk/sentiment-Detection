#!/usr/bin/env python
# coding: utf-8

# # Step 1 : import Required Packages

# In[2]:


import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
import pyodbc
import nltk
from nltk.stem import PorterStemmer
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import time
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
import pickle
from pandas.core.frame import DataFrame
from sqlalchemy import create_engine
import urllib
from mstranslator import Translator
from langdetect import detect
from langdetect import DetectorFactory
DetectorFactory.seed = 0
translator = Translator('9c59b894896642b8ba8810549f0602fc')


# #Step 2: Import required Tables

# In[9]:

print("start")
conn = pyodbc.connect('Trusted_Connection=yes;'
                      'Driver={SQL Server};'
                      'Server=KSRVMSQLSTG115\MERISQLSTGIDB;'
                      'Database=DQS_STAGING_DATA;')

Questions = """SELECT * FROM [DQS_STAGING_DATA].[dbo].[DSIB_T_QUESTIONS]"""
Questions = pd.read_sql_query(Questions , conn)

score = """SELECT d.[USERNAME]
      ,round(avg(a.Scoring),0) Scoring
  FROM [DQS_STAGING_DATA].[dbo].[DSIB_TRX_ANSWERS_DETAIL] d
  inner join DSIB_T_QUESTIONS_ANSWEROPTIONS a on d.year =a.year and d.QID = a.QuestionID and a.ID = d.AnswerID
  where a.Flag=1 
  group by d.USERNAME"""
score = pd.read_sql_query(score , conn)

Answers_raw = """SELECT USERNAME,AnswerID FROM [DQS_STAGING_DATA].[dbo].[DSIB_TRX_ANSWERS_DETAIL]"""
Answers = pd.read_sql_query(Answers_raw , conn)

sqlstr = """select  [ID] as unique_key ,[Year],[QID],[AnswerText],[USERNAME],[IsAnalyzed] from DSIB_TRX_ANSWERS_DETAIL where AnswerText !='' and IsAnalyzed != 1 """ 
#sqlstr = """select [ID] as unique_key ,[Year],[QID],[AnswerText],[USERNAME],[IsAnalyzed] from DSIB_TRX_ANSWERS_DETAIL where AnswerText !=''""" 
data_comments = pd.read_sql_query(sqlstr, conn)

Grouping = """SELECT  * FROM Groups_WordCloud"""
Grouping = pd.read_sql_query(Grouping , conn)

tr_sql1 = """SELECT  * FROM [DQS_STAGING_DATA].[dbo].[TrainingData_Sentiment]"""
#Comments file is the training data
Comments = pd.read_sql_query(tr_sql1 , conn)
Comments['AnswerText'] = Comments['AnswerText'].astype(str)
Comments['AnswerText'] = Comments['AnswerText'].dropna()

tr_sql2 = """select [ID] as unique_key,[Year],[QID],[AnswerText],[USERNAME] from DSIB_TRX_ANSWERS_DETAIL where AnswerText !='' and [IsAnalyzed] != 1"""
#tr_sql2 = """select [ID] as unique_key,[Year],[QID],[AnswerText],[USERNAME] from DSIB_TRX_ANSWERS_DETAIL where AnswerText !='' """

data = pd.read_sql_query(tr_sql2 , conn)
#data file is the data to analyse
data['AnswerText'] = data['AnswerText'].astype(str)
data['AnswerText'] = data['AnswerText'].dropna()

tr_sql3 = """SELECT  * FROM [DQS_STAGING_DATA].[dbo].[Emotions_Sentiment]"""
emotions = pd.read_sql_query(tr_sql3 , conn)
#In emotions file almost 3k words are tagged to 8 emotions
data['AnswerText'] = data['AnswerText'].astype(str)
data['AnswerText'] = data['AnswerText'].str.lower()

print("import done")

# # step 3 : defining custom Functions

# In[4]:


def remove_numbers(sentence):
    sentence = re.sub("\S*\d\S*", " ", sentence).strip()
    return (sentence)
def removePunctuations(sentence):
    cleaned_text  = re.sub('[^a-zA-Z]',' ',sentence)
    return (cleaned_text)
#remove words containing leters repeating more 3 times
def removePatterns(sentence): 
    cleaned_text  = re.sub("\\s*\\b(?=\\w*(\\w)\\1{2,})\\w*\\b",' ',sentence)
    return (cleaned_text)
def emotion_detector(string,word_list):
    splitted_text = string.split(' ')
    mylist = list(word_list['word'])
    allmatch = [x for x in  mylist if x in splitted_text]
    without_dups = list(set(allmatch))
    y = ','.join(without_dups)
    return y
def string_length(text):
    if text == '':
        return 0
    else:
        count = len(text.split(','))
        return count
def group(full,sub):
    return full.find(sub)


# # Step 4 : Sentiment Analysis

# In[5]:


##Step 4.1 : Text preprocessing for Training Data 
porter=PorterStemmer()
string=' '    
stemed_word=' '
preprocessed_clean_data =[]

for clean_data in Comments['AnswerText']:
    filtered_sentence=[]
    clean_data = remove_numbers(clean_data)
    clean_data = removePunctuations(clean_data)
    clean_data = removePatterns(clean_data)
        
    for cleaned_words in clean_data.split():
        if len(cleaned_words)>1:
            stemed_word=(porter.stem(cleaned_words.lower()))                                   
            filtered_sentence.append(stemed_word)
        else:
            continue
    clean_data = " ".join(filtered_sentence)
    preprocessed_clean_data.append(clean_data.strip())

Comments['Clean_text'] = preprocessed_clean_data

# In[6]:


##step 4.2 : Build Sentiment Analysis Model
vectorizer = TfidfVectorizer( smooth_idf=True, norm="l2", 
                             sublinear_tf=False, ngram_range=(1,3))
tfidf = vectorizer.fit_transform(Comments.Clean_text)

classifier = LinearSVC( penalty='l2',dual=False, tol=1e-6,class_weight='balanced',max_iter = 3000)
classifier =CalibratedClassifierCV(classifier)
classifier.fit(tfidf, Comments['Emotions'])


# In[10]:


##Step 4.3: Save and load the model
filename = 'C:/Users/msqlcladmin/industry_classification_linearSVM_new.sav'
pickle.dump(classifier, open(filename, 'wb'), protocol=2)

filename1 = 'C:/Users/msqlcladmin/vectorizer.sav'
pickle.dump(vectorizer, open(filename1, 'wb'), protocol=2)

filename = 'C:/Users/msqlcladmin/industry_classification_linearSVM_new.sav'
file = str(filename)
Svm_model = pickle.load(open(file, 'rb'))

filename1 = 'C:/Users/msqlcladmin/vectorizer.sav'
file = str(filename1)
vectorizer = pickle.load(open(file, 'rb'))


# In[11]:


##Step 4.4 : Text preprocessing and Prediction for actual data 
porter=PorterStemmer()
string=' '    
stemed_word=' '
preprocessed_clean_data =[]

for clean_data in data['AnswerText']:
    filtered_sentence=[]
    clean_data = remove_numbers(clean_data)
    clean_data = removePatterns(clean_data)
    s = detect(clean_data)
    if s !='en':
        clean_data = translator.translate(clean_data, lang_from ='ar', lang_to='en')
    else:
        clean_data = removePunctuations(clean_data)
        
    for cleaned_words in clean_data.split():
        if len(cleaned_words)>1:
            stemed_word=(porter.stem(cleaned_words.lower()))                                   
            filtered_sentence.append(stemed_word)
        else:
            continue
    clean_data = " ".join(filtered_sentence)
    preprocessed_clean_data.append(clean_data.strip())

X_test = vectorizer.transform(data.Clean_text)

data.drop(['Clean_text'],axis = 1,inplace=True)

predict_svm = Svm_model.predict(X_test)
data['Sentiment'] = predict_svm
probability = Svm_model.predict_proba(X_test)
probability = pd.DataFrame(probability)
probability.columns = ['one','two','three']
probability['max'] = probability[['one','two','three']].max(axis = 1)
data['Probability'] = round(probability['max']*100,2)
data=data.rename(columns = {'USERNAME':'USER NAME'})


## Step 5: Detecting Emotions 

# In[12]:


start = time.time()
print(start)
data['trust_words'] = data.apply(lambda x: emotion_detector(x['AnswerText'],emotions[emotions['sentiment']=='trust']),axis = 1)
data['surprise_words'] = data.apply(lambda x: emotion_detector(x['AnswerText'],emotions[emotions['sentiment']=='surprise']),axis = 1)
data['joy_words'] = data.apply(lambda x: emotion_detector(x['AnswerText'],emotions[emotions['sentiment']=='joy']),axis = 1)
data['anticipation_words'] = data.apply(lambda x: emotion_detector(x['AnswerText'],emotions[emotions['sentiment']=='anticipation']),axis = 1)
data['fear_words'] = data.apply(lambda x: emotion_detector(x['AnswerText'],emotions[emotions['sentiment']=='fear']),axis = 1)
data['sadness_words'] = data.apply(lambda x: emotion_detector(x['AnswerText'],emotions[emotions['sentiment']=='sadness']),axis = 1)
data['anger_words'] = data.apply(lambda x: emotion_detector(x['AnswerText'],emotions[emotions['sentiment']=='anger']),axis = 1)
data['disgust_words'] = data.apply(lambda x: emotion_detector(x['AnswerText'],emotions[emotions['sentiment']=='disgust']),axis = 1)


data['trust_words'] = np.where(data['Sentiment']!='Negative',data['trust_words'],'')
data['surprise_words'] = np.where(data['Sentiment']!='Negative',data['surprise_words'],'')
data['joy_words'] = np.where(data['Sentiment']!='Negative',data['joy_words'],'')
data['anticipation_words'] = np.where(data['Sentiment']!='Negative',data['anticipation_words'],'')
data['fear_words'] = np.where(data['Sentiment']!='Positive',data['fear_words'],'')
data['sadness_words'] = np.where(data['Sentiment']!='Positive',data['sadness_words'],'')
data['anger_words'] = np.where(data['Sentiment']!='Positive',data['anger_words'],'')
data['disgust_words'] = np.where(data['Sentiment']!='Positive',data['disgust_words'],'')

data['count_trust'] = data['trust_words'].apply(lambda x: string_length(x))
data['count_fear'] = data['fear_words'].apply(lambda x: string_length(x))
data['count_sadness'] = data['sadness_words'].apply(lambda x: string_length(x))
data['count_anger'] = data['anger_words'].apply(lambda x: string_length(x))
data['count_surprice'] = data['surprise_words'].apply(lambda x: string_length(x))
data['count_disgust'] = data['disgust_words'].apply(lambda x: string_length(x))
data['count_joy'] = data['joy_words'].apply(lambda x: string_length(x))
data['count_anticipation'] = data['anticipation_words'].apply(lambda x: string_length(x))
print(time.time())
print(time.time() - start)





# ## Step 7:Grouping by keywords

# In[16]:


data_comments['join']=1
Grouping['join']=1
Output = data_comments.merge(Grouping,on = 'join').drop(['join'],axis=1)
Output['count']= Output.apply(lambda x: group(x['AnswerText'],x['word']),axis = 1)
Output['group'] = np.where(Output['count']>0,Output['Grouping'],"Others")
Output.drop(['count','word','Grouping'],axis=1,inplace =True)
Output.drop_duplicates(inplace =True)
Output['flag'] = np.where(Output['group']=='Others',2,1)
Output.sort_values(by = ['unique_key','flag'],inplace =True,ascending=True)
Output.drop_duplicates(['unique_key'],inplace = True)


# In[18]:


string=' '    
stemed_word=' '
preprocessed_clean_data =[]
stop_words = set(stopwords.words('english'))
custom_words = ['','comment','nil','NA','na','asf','xzx','ljk','NA ','kb','kjji','kk','fuck','-',
                           'ljk','lmnon','Fuck','-NIL-','zxz','Znfndjzjw3e','zedbazi','////','no','!']
from nltk.stem import WordNetLemmatizer 
lemmatizer = WordNetLemmatizer()
for clean_data in Output['AnswerText']:
    filtered_sentence=[]
    lematised_text = []
    clean_data = remove_numbers(clean_data)
    clean_data = removePunctuations(clean_data)
    clean_data = removePatterns(clean_data)
    
    for cleaned_words in clean_data.split():
        if len(cleaned_words)>1:
            stemed_word=(lemmatizer.lemmatize(cleaned_words.lower(),pos = 'a'))                                   
            filtered_sentence.append(stemed_word)
        else:
            continue
    clean_data = " ".join(filtered_sentence)
    preprocessed_clean_data.append(clean_data.strip())

Output['wordcloud'] = preprocessed_clean_data
Output['wordcloud']=Output['wordcloud'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop_words)]))
Output['wordcloud']=Output['wordcloud'].apply(lambda x: ' '.join([w for w in x.split() if w not in (custom_words)]))


# In[43]:


Final_file = pd.merge(Output,score,how = 'left',on = 'USERNAME')
Final_file.drop(['Year','QID','AnswerText','flag'],inplace=True,axis =1)
Final_file1 = pd.merge(Final_file,data,on = 'unique_key')
Final_file1.rename(columns={'group':'Grouping'},inplace=True)

print("end")

# # Step 8 : Uploading the results as a sepearate table in sql server
# In[46]:

cursor = conn.cursor()
cursor.execute('UPDATE [DSIB_TRX_ANSWERS_DETAIL] SET IsAnalyzed = 1')

quoted = urllib.parse.quote_plus('DRIVER={SQL Server};SERVER=KSRVMSQLSTG115\MERISQLSTGIDB;DATABASE=DQS_STAGING_DATA;Trusted_Connection=yes;')
engine = create_engine('mssql+pyodbc:///?odbc_connect={}'.format(quoted))
Final_file1.to_sql(name = 'DSIB_Sentiement_Scoring_wordcloud', con=engine, schema='dbo', if_exists='append', index = False)

#print('finished')
