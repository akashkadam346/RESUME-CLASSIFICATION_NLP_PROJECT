#!/usr/bin/env python
# coding: utf-8

# ### IMPORT NECESSARY LIBRARIES

# In[1]:


import docx
import glob
import warnings
warnings.filterwarnings("ignore")
import spacy
import pickle
import random
from spacy import displacy
from spacy import schemas
from spacy import Dict
from spacy import displacy
from spacy.lang.en.stop_words import  STOP_WORDS
import string
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import textract
import antiword
import matplotlib.pyplot as plt
from PyPDF2 import PdfFileReader
import re
import nltk
from nltk.corpus import stopwords
stop = stopwords.words('english')
#nltk.download('punkt')
#nltk.download('averaged_perceptron_tagger')
#nltk.download('maxent_ne_chunker')
#nltk.download('words')
from spacy.matcher import Matcher
#nltk.download('stopwords')
#nltk.download('wordnet')

from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
nlp = spacy.load("en_core_web_sm")


# ### IMPORT DATASETS

# In[2]:


data1 = pd.read_csv("C:/Users/MIZNA/Downloads/intership_resumes.csv")
data2 = pd.read_csv("C:/Users/MIZNA/Downloads/Peoplesoft_Resumes.csv")
data3 = pd.read_csv("C:/Users/MIZNA/Downloads/SQLDeveloperLightning_Resumes.csv")
data4 = pd.read_csv("C:/Users/MIZNA/Downloads/workday_resumes.csv")
data5 = pd.read_csv("C:/Users/MIZNA/Downloads/React_Developer_resumes.csv")


# In[3]:


Resume = pd.concat([data1,data2,data3,data4,data5],axis=0)
Resume = Resume.reset_index()
Resume = Resume.drop(columns='Number',axis=0)
Resume = Resume.drop(columns='index',axis=0)
Resume


# ### Exploratory Data Analysis (EDA)

# In[4]:


Resume.info()


# In[5]:


Resume.isnull().sum()


# ### Data Preprocessing

# ##### We will perform label encoding to convert category variable from string datatype to float datatype

# In[6]:


from sklearn.preprocessing import LabelEncoder
le_encoder = LabelEncoder()
Resume["Encoded_Skill"] = le_encoder.fit_transform(Resume["Label"])
Resume.head()


# In[7]:


Resume.Label.value_counts()


# In[8]:


print("Displaying the distinct categories of resume -")
print(Resume.Label.unique())


# ### Data Cleaning

# In[9]:


import re #REGULAR EXPRESSION
import string

def clean_text(CV):
    '''Make text lowercase, remove text in square brackets, remove punctuation and remove words containing numbers.'''
    CV = CV.lower()
    CV = re.sub('\[.*?\]', '', CV)
    CV = re.sub('[%s]' % re.escape(string.punctuation), '', CV)
    CV = re.sub('\w*\d\w*', '', CV)
    CV = re.sub("[0-9" "]+"," ",CV)
    CV = re.sub('[‘’“”…]', '', CV)
    return CV

clean = lambda x: clean_text(x)


# In[10]:


Resume['CV'] = Resume.CV.apply(clean)
Resume.CV


# ### Word frequency BEFORE removal of STOPWORDS

# In[11]:


#Word Frequency
frequency = pd.Series(' '.join(Resume['CV']).split()).value_counts()[:20] #For top 20
frequency


# ### Removing STOPWORDS

# In[12]:


from nltk.corpus import stopwords
stop = stopwords.words('english')
Resume['CV'] = Resume['CV'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))


# ### Word frequency AFTER removal of STOPWORDS

# In[13]:


frequency_Sw = pd.Series(' '.join(Resume['CV']).split()).value_counts()[:20] # for top 20
frequency_Sw


# ### Performing A NER (Using Spacy)

# In[14]:


nlp = spacy.load("en_core_web_sm")
text=nlp(Resume["CV"][0])
displacy.render(text, style = "ent")


# ### VISUALIZATION OF DATASET

# In[15]:


import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use('seaborn-dark-palette')
plt.figure(figsize=(15,7))
plt.title("The distinct categories of resumes")
plt.xticks(rotation=90)
sns.countplot(y="Label", data=Resume,palette=("Set2"))
plt.show()


# In[16]:


from matplotlib.gridspec import GridSpec
targetCounts = Resume.Label.value_counts()
targetLabels  = Resume.Label.unique()
# Make square figures and axes
plt.figure(1, figsize=(25,25))
the_grid = GridSpec(2, 2)


cmap = plt.get_cmap('plasma')
colors = ['#ff9999','#66b3ff','#99ff99','#ffcc99','#c2c2f0']
plt.subplot(the_grid[0, 1], aspect=1, title='CATEGORY DISTRIBUTION')


source_pie = plt.pie(targetCounts, labels=targetLabels, autopct='%1.1f%%', shadow=True, colors=colors)
plt.show()


# #### First take a look at the number of Characters present in each sentence. This can give us a rough idea about the resume length

# ##### Calculating each Characterstic in dataframe

# In[17]:


characters=Resume["CV"].apply(len)
characters


# In[18]:


print('Total Number of characters dataset:',characters.sum())
print('Mean of each characters in datset:',characters.mean())
print('Median of characters in dataset:',characters.median())
print('Standard Deviation of characters in dataset:',characters.std())
print('skew of characters dataset:',characters.skew())


# In[19]:


characters.describe()


# In[20]:


import seaborn as sns
import matplotlib.pyplot as plt
sns.distplot(x = characters)


# #### Calculating each Word Characterstic in dataframe

# In[21]:


words = Resume['CV'].apply(lambda x: len(str(x).split(' ')))
words


# In[22]:


print('Total Number of Word in dataset:',words.sum())
print('Mean of each Word in datset:',words.mean())
print('Median of Word in dataset:',words.median())
print('Standard Deviation of Word in dataset:',words.std())
print('skew of Word dataset:',words.skew())


# In[23]:


sns.distplot(x = words)


# In[ ]:





# #### Feature Extraction

# In[24]:


from collections import Counter
import seaborn as sns


# In[25]:


words =['using','Workday','Experience','PeopleSoft',
 'experience','SQL','Application','data','Server',
 'business','Project','reports','like','HCM','Worked',
 'knowledge','Involved','various','Good', 'Reports','React','EIB','integrations','Web','system','creating','issues',
 'Created', 'Responsibilities','Process','process','support', 
 'application','new','People','I','team','working', 
 'Database','database','Integration','Domains','client', 
 'requirements','Core',  'Business', 
'Oracle','Report', 'Developer', 'Data']
indices = np.random.zipf(1.6, size=500).astype(np.int) % len(words)
tw = np.array(words)[indices]

tf = Counter(tw)

y = [count for tag, count in tf.most_common(50)]
x = [tag for tag, count in tf.most_common(50)]
plt.style.use('seaborn-dark-palette')
plt.figure(figsize=(12,5))
plt.bar(x, y, color=['gold','lightcoral', 'lightskyblue'])
plt.title("Word frequencies in Resume Data in Log Scale")
plt.ylabel("Frequency (log scale)")
plt.yscale('symlog') # optionally set a log scale for the y-axis
plt.xticks(rotation=90)
for i, (tag, count) in enumerate(tf.most_common(50)):
    plt.text(i, count, f' {count} ', rotation=90,
             ha='center', va='top' if i < 10 else 'bottom', color='white' if i < 10 else 'black')
plt.xlim(-0.6, len(x)-0.4) # optionally set tighter x lims
plt.tight_layout() # change the whitespace such that all labels fit nicely
plt.show()


# In[26]:


def wordBarGraphFunction_1(df,column,title):
    topic_words = [ z.lower() for y in
                       [ x.split() for x in df[column] if isinstance(x, str)]
                       for z in y]
    word_count_dict = dict(Counter(topic_words))
    popular_words = sorted(word_count_dict, key = word_count_dict.get, reverse = True)
    popular_words_nonstop = [w for w in popular_words if w not in stopwords.words("english")]
    plt.style.use('fivethirtyeight')
    sns.barplot(x=np.arange(20),y= [word_count_dict[w] for w in reversed(popular_words_nonstop[0:20])])
    plt.xticks([x + 0.5 for x in range(20)], reversed(popular_words_nonstop[0:20]),rotation=90)
    plt.title(title)
    plt.show()


# In[27]:


plt.figure(figsize=(15,6))
wordBarGraphFunction_1(Resume,"CV","Most frequent Words ")


# #### WORDCLOUD

# In[28]:


# Import packages
import matplotlib.pyplot as plt
from IPython import get_ipython
get_ipython().run_line_magic('inline')
from wordcloud import WordCloud, STOPWORDS
# Define a function to plot word cloud
def plot_cloud(wordcloud):
    # Set figure size
    plt.figure(figsize=(40, 30))
    # Display image
    plt.imshow(wordcloud) 
    # No axis details
    plt.axis("off");


# In[29]:


# Generate wordcloud
stopwords = STOPWORDS
stopwords.add('will')
wordcloud = WordCloud(width = 3000, height = 2000, background_color='black', max_words=100,colormap='Set2',stopwords=stopwords).generate(str(Resume))
# Plot
plot_cloud(wordcloud)


# #### Bag Of Words

# In[30]:


requiredText = Resume["CV"]
requiredTarget = Resume["Encoded_Skill"].values
Countvectorizer=CountVectorizer(analyzer='word',token_pattern=r'\w{1,}',stop_words = 'english')
bag = Countvectorizer.fit_transform(requiredText)
Countvectorizer.vocabulary_


# #### VECTORIZATION

# #### COUNT VECTORIZER tells the frequency of a word.

# In[31]:


vectorizer1 = CountVectorizer(min_df = 1, max_df = 0.9)
count_vect = vectorizer1.fit_transform(Resume["CV"])
word_freq_df = pd.DataFrame({'term': vectorizer1.get_feature_names(), 'occurrences':np.asarray(count_vect.sum(axis=0)).ravel().tolist()})
word_freq_df['frequency'] = word_freq_df['occurrences']/np.sum(word_freq_df['occurrences'])
word_freq_df


# In[32]:


sns.distplot(x =[word_freq_df['frequency']])


# #### TFIDF - Term frequency inverse Document Frequency

# In[33]:


from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# In[34]:


word_vectorizer = TfidfVectorizer(sublinear_tf=True, stop_words='english',max_features=1500)
word_vectorizer.fit(requiredText)
WordFeatures = word_vectorizer.transform(requiredText)


# ### Model Building || Model Training || Model Evaluation

# #### DATA PREPARATION

# In[35]:


x_train,x_test,y_train,y_test = train_test_split(WordFeatures, requiredTarget, random_state=0, test_size=0.2)
print("X Train shape:",x_train.shape)
print("Y Train shape:",y_train.shape)
print("x Test shape:",x_test.shape)
print("y Test shape:",y_test.shape)


# ### 1. LOGISTIC REGRESSION

# In[36]:


#IMPORTING NECESSARY LIBRARIES FOR LOGISTIC REGRESSION
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import confusion_matrix,roc_auc_score,accuracy_score,precision_score,recall_score,f1_score,classification_report


# In[37]:


logistic_classifier = LogisticRegression()
logistic_classifier.fit(x_train,y_train)

#Predicting on Training Data
pred_train_log = logistic_classifier.predict(x_train)
#Accuracy On Train Data
train_acc_log = np.mean(pred_train_log==y_train)
print("ACCURACY OF TRAIN DATA IN LOGISTIC REGRESSION:", train_acc_log)

#Predicting on Test Data
pred_test_log = logistic_classifier.predict(x_test)
#Accuracy On Test Data
test_acc_log = np.mean(pred_test_log==y_test)
print("ACCURACY OF TEST DATA IN LOGISTIC REGRESSION:",test_acc_log )

#Confusion Matrix
logistic_cm = confusion_matrix(y_test,pred_test_log)

#Classification Report
print("CLASSIFICATION REPORT OF LOGISTIC REGRESSION:\n", classification_report(y_test,pred_test_log))


# In[38]:


accuracy_log = round(accuracy_score(y_test,pred_test_log),4)
precision_log = round(precision_score(y_test,pred_test_log,average = 'macro'),4)
recall_log = round(recall_score(y_test,pred_test_log,average = 'macro'),4)
f1_log = round(f1_score(y_test,pred_test_log,average = 'macro'),4)

#Printing Accuracy, Recall, precision, F1_score
print('Accuracy Score   : ',accuracy_log )
print('Precision Score  : ',precision_log )
print('Recall Score     : ', recall_log)
print('f1-Score         : ',f1_log )


# In[ ]:





# ### 2. DECISION TREE

# In[39]:


#IMPORTING NECESSARY LIBRARIES FOR DECISION TREE
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier


# In[40]:


DT = DecisionTreeClassifier()
DT_classifier = DecisionTreeClassifier(criterion = 'entropy', max_depth=2)
DT_classifier.fit(x_train,y_train)

#Predicting on Train Data
pred_train_dt = DT_classifier.predict(x_train)
#Accuracy On Train Data
train_acc_dt = np.mean(pred_train_dt==y_train)
print("ACCURACY OF TRAIN DATA IN DECISION TREE:",train_acc_dt )

#Predicting on Test Data
pred_test_dt = DT_classifier.predict(x_test)
#Accuracy on Test Data
test_acc_dt = np.mean(pred_test_dt==y_test)
print("ACCURACY OF TEST DATA IN DECISION TREE:",test_acc_dt )

#Confusion Matrix
dt_cm = confusion_matrix(y_test,pred_test_dt)

#Classification Report
print("CLASSIFICATION REPORT OF DECISION TREE:\n", classification_report(y_test,pred_test_dt))


# In[41]:


accuracy_dt = round(accuracy_score(y_test,pred_test_dt),4)
precision_dt = round(precision_score(y_test,pred_test_dt,average = 'macro'),4)
recall_dt = round(recall_score(y_test,pred_test_dt,average = 'macro'),4)
f1_dt = round(f1_score(y_test,pred_test_dt,average = 'macro'),4)

#Printing Accuracy, Recall, precision, F1_score
print('Accuracy Score   : ',accuracy_dt )
print('Precision Score  : ',precision_dt )
print('Recall Score     : ', recall_dt)
print('f1-Score         : ',f1_dt )


# ### 3. RANDOM FOREST 

# In[42]:


#IMPORTING NECESSARY LIBRARIES FOR RANDOM FOREST
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier


# In[43]:


RF = {'n_estimators':15,'class_weight': "balanced",'n_jobs':-1,'random_state':42}
RF_classifier = RandomForestClassifier(**RF)
RF_classifier.fit(x_train,y_train)

#Predicting on Train Data
pred_train_rf = RF_classifier.predict(x_train)
#Accuracy On Train Data
train_acc_rf = np.mean(pred_train_rf==y_train)
print("ACCURACY OF TRAIN DATA IN RANDOM FOREST:",train_acc_rf)

#Predicting on Test Data
pred_test_rf = RF_classifier.predict(x_test)
#Accuracy On Test Data
test_acc_rf = np.mean(pred_test_rf==y_test)
print("ACCURACY OF TEST DATA IN RANDOM FOREST:",test_acc_rf )

#Confusion Matrix
rf_cm = confusion_matrix(y_test,pred_test_rf)

#Classification Report
print("CLASSIFICATION REPORT OF RANDOM FOREST:\n", classification_report(y_test,pred_test_rf))


# In[44]:


accuracy_rf = round(accuracy_score(y_test,pred_test_rf),4)
precision_rf = round(precision_score(y_test,pred_test_rf,average = 'macro'),4)
recall_rf = round(recall_score(y_test,pred_test_rf,average = 'macro'),4)
f1_rf = round(f1_score(y_test,pred_test_rf,average = 'macro'),4)

#Printing Accuracy, Recall, precision, F1_score
print('Accuracy Score   : ',accuracy_rf )
print('Precision Score  : ',precision_rf )
print('Recall Score     : ', recall_rf)
print('f1-Score         : ',f1_rf )


# ### 4. MULTINOMIAL NAVIE BAYES

# In[45]:


#IMPORTING NECESSARY LIBRARIES FOR MULTINOMIAL NAVIE BAYES
from sklearn.naive_bayes import MultinomialNB as MB


# In[46]:


classifier_mb = MB()
classifier_mb.fit(x_train,y_train)

#Predicting On Train Data
pred_train_mb = classifier_mb.predict(x_train)
#Accuracy On Train Data
train_acc_mb = np.mean(pred_train_mb==y_train)
print("ACCURACY OF TRAIN DATA IN MULTINOMIAL NAVIE BAYES:", train_acc_mb)

#Predicting On Test Data
pred_test_mb = classifier_mb.predict(x_test)
#Accuracy On Test Data
test_acc_mb = np.mean(pred_test_mb==y_test)
print("ACCURACY OF TEST DATA IN MULTINOMIAL NAVIE BAYES:", test_acc_mb)

#Confusion Matrix
mb_cm = confusion_matrix(y_test,pred_test_mb)

#Classification Report
print("CLASSIFICATION REPORT OF MULTINOMIAL NAVIE BAYES:\n", classification_report(y_test,pred_test_mb))


# In[47]:


accuracy_mb = round(accuracy_score(y_test,pred_test_mb),4)
precision_mb = round(precision_score(y_test,pred_test_mb,average = 'macro'),4)
recall_mb = round(recall_score(y_test,pred_test_mb,average = 'macro'),4)
f1_mb = round(f1_score(y_test,pred_test_mb,average = 'macro'),4)

#Printing Accuracy, Recall, precision, F1_score
print('Accuracy Score   : ',accuracy_mb )
print('Precision Score  : ',precision_mb )
print('Recall Score     : ', recall_mb)
print('f1-Score         : ',f1_mb )


# ### 5. SUPPORT VECTOR MACHINE 

# In[48]:


##IMPORTING NECESSARY LIBRARIES FOR SUPPORT VECTOR MACHINE
from sklearn import svm
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV


# In[49]:


svm_classifier = (SVC(kernel='linear'))
svm_classifier.fit(x_train,y_train)

#Predicting On Train Data
pred_train_svm = svm_classifier.predict(x_train)
#Accuracy On Train Data
train_acc_svm = np.mean(pred_train_svm==y_train)
print("ACCURACY OF TRAIN DATA IN SUPPORT VECTOR MACHINE:",train_acc_svm )

#Prediciting On Test Data
pred_test_svm = svm_classifier.predict(x_test)
#Accuracy On Test Data
test_acc_svm = np.mean(pred_test_svm==y_test)
print("ACCURACY OF TEST DATA IN SUPPORT VECTOR MACHINE:",test_acc_svm)

#Confusion Matrix
svm_cm = confusion_matrix(y_test,pred_test_svm)

#Classification Report
print("CLASSIFICATION REPORT OF SUPPORT VECTOR MACHINE:\n", classification_report(y_test,pred_test_svm))


# In[50]:


accuracy_svm = round(accuracy_score(y_test,pred_test_svm),4)
precision_svm = round(precision_score(y_test,pred_test_svm,average = 'macro'),4)
recall_svm = round(recall_score(y_test,pred_test_svm,average = 'macro'),4)
f1_svm = round(f1_score(y_test,pred_test_svm,average = 'macro'),4)

#Printing Accuracy, Recall, precision, F1_score
print('Accuracy Score   : ',accuracy_svm )
print('Precision Score  : ',precision_svm )
print('Recall Score     : ', recall_svm)
print('f1-Score         : ',f1_svm )


# ### CONFUSION MATRIX

# In[51]:


plt.figure(figsize=(20,15))

plt.suptitle("Confusion Matrixes", fontsize=18)

plt.subplot(2,3,1)
plt.title("LOGISTIC REGRESSION")
sns.heatmap(logistic_cm, cbar=False, annot=True, cmap="mako",  fmt="d")

plt.subplot(2,3,2)
plt.title("DECISION TREE")
sns.heatmap(dt_cm, cbar=False, annot=True, cmap="Blues", fmt="d")

plt.subplot(2,3,3)
plt.title("RANDOM FOREST CLASSIFICATION")
sns.heatmap(rf_cm, cbar=False, annot=True, cmap="BuPu", fmt="d")

plt.subplot(2,3,4)
plt.title("NaiveBayes Classification")
sns.heatmap(mb_cm, cbar=False, annot=True, cmap="Greens", fmt="d")

plt.subplot(2,3,5)
plt.title("SVM Classification")
sns.heatmap(svm_cm, cbar=False, annot=True, cmap="YlGnBu",  fmt="d")

plt.show()


# In[52]:


table = {'Classifier' : ['LOGISTIC REGRESSION', 'DECISION TREE', 'RANDOM FOREST', 'MULTINOMIAL NAIVE BAYES', 'SUPPORT VECTOR MACHINE'], 'Accuracy_Score' : [accuracy_log, accuracy_dt, accuracy_rf, accuracy_mb, accuracy_svm], 'Precision_Score' : [precision_log, precision_dt, precision_rf, precision_mb, precision_svm], 'Recall_Score' : [recall_log, recall_dt, recall_rf, recall_mb, recall_svm], 'F1-Score' : [f1_log, f1_dt, f1_rf, f1_mb, f1_svm]}
table = pd.DataFrame(table)
table


# ### ACCURACY COMPARISON PLOT

# In[53]:


#Accuracy
plt.figure(figsize=(15,6))
ax= sns.barplot(x=table.Classifier, y=table.Accuracy_Score, palette =sns.color_palette("Set2") )
ax.set_xticklabels(ax.get_xticklabels(),rotation=30)
plt.xlabel('Classification Models')
plt.ylabel('Accuracy')
plt.title('Accuracy Scores of Classification Models')
for i in ax.patches:
    ax.text(i.get_x()+.19, i.get_height()-0.3,             str(round((i.get_height()), 4)), fontsize=15, color='b')
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:




