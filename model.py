import pandas as pd
import numpy as np
import requests
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle


# Data Extraction
url = 'https://raw.githubusercontent.com/laxmimerit/All-CSV-ML-Data-Files-Download/master/spam.tsv'
result = requests.get(url,allow_redirects=True)
with open('spam_data.tsv','wb') as file:
    file.write(result.content)

data = pd.read_csv('spam_data.tsv',sep='\t')

# preview the Data
# data.head()

# data['label'].value_counts(); checks to see the distribution of each class
ham = data[data['label']=='ham']
spam = data[data['label']=='spam']

ham = ham.sample(spam.shape[0])

# ham.shape, spam.shape; this checks to see the distribution of each class
data = ham.append(spam,ignore_index=True)

# creating training and testing Data
X_train,X_test,y_train,y_test = train_test_split(data['message'],data['label'],test_size=0.3,random_state=0,shuffle=True,stratify=data['label'])

clf = Pipeline([('tfidf',TfidfVectorizer()),('rf',RandomForestClassifier(n_estimators=100,n_jobs=1))])
clf.fit(X_train,y_train)

# Make pickle file
pickle.dump(clf, open("model.pkl","wb"))
