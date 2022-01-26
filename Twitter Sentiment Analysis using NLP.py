# -*- coding: utf-8 -*-
"""
Created on Sun Jan 23 20:49:50 2022

@author: Priyam Ghosh
"""

""" Twitter Sentiment Analysis using Natural Language Processing """

import pandas as pd
import numpy as np
import re #Regular Expression
import nltk
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, plot_roc_curve, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import os

os.chdir(os.getcwd()+"\\Desktop\\Twitter Sentiment Dataset")

df=pd.read_csv("dataset.csv")
df.shape

features=df.iloc[:,10].values
labels=df.iloc[:,1].values

#Removing the special character
processed_features=[]
for sentence in range(0,len(features)):
    #remove all the special characters
    processed_feature=re.sub(r'\W',' ',str(features[sentence]))
    
    #remove all single characters
    processed_feature=re.sub(r'\s+[a-zA-Z]\s+',' ',processed_feature)
    
    #remove single characters from the start
    processed_feature=re.sub(r'\^[a-zA-Z]\s+',' ',processed_feature)
    
    #substituting multiple spaces with single spaces
    processed_feature=re.sub(r'\s+',' ',processed_feature,flags=re.I)
    
    #removing prefixed 'b'
    processed_feature=re.sub(r'^b\s+','',processed_feature)
    
    #converting to lowercase
    processed_feature=processed_feature.lower()
    
    processed_features.append(processed_feature)
    
#Feature Extraction from text
nltk.download('stopwords')
vectorizer = TfidfVectorizer(max_features=2500, min_df=7, max_df=0.8, stop_words=stopwords.words('english'))
processed_features=vectorizer.fit_transform(processed_features).toarray()

#splitting dataset into train and test
X_train,X_test,y_train,y_test=train_test_split(processed_features,labels,test_size=0.2,random_state=0)

#loading Model
rfc=RandomForestClassifier(n_estimators=200,random_state=0)
rfc.fit(X_train,y_train)
prediction=rfc.predict(X_test)


#Confusion Matrix
from sklearn import metrics
import itertools
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

cm = metrics.confusion_matrix(y_test, prediction, labels=['negative', 'neutral', 'positive'])
plot_confusion_matrix(cm, classes=['negative', 'neutral', 'positive'])


print(accuracy_score(y_test,prediction))



ada=AdaBoostClassifier(base_estimator=rfc)
ada.fit(X_train,y_train)

y_predict=ada.predict(X_test)

#plotting ROC curve
print('score: ', ada.score)

metrics.plot_roc_curve(ada,X_test,y_test)
plt.show()


param_dist={'n_estimators':[40,50,60,70,80], 'learning_rate':[0.04,0.03,0.02,0.1],'algorithm':['SAMME', 'SAMME.R']}
grid_1=RandomizedSearchCV(ada,param_distributions=param_dist,cv=5,n_jobs=-1)
grid_1.fit(X_train,y_train)

#plotting ROC curve
print('score: ', grid_1.best_score_)
print()
print('ROC-AUC curve')
metrics.plot_roc_curve(grid_1,X_test,y_test)
plt.show()

predict=grid_1.predict(X_test)
accuracy_score(y_test, predict)
print('classification report')
print(classification_report(y_test,predict))

boosted_model=grid_1.best_estimator_
print('confusion_matrix')
print(confusion_matrix(y_test,predict))


























































