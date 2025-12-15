import sklearn.preprocessing as sp
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import sklearn.impute as si

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sb
import pickle

from sklearn.metrics import accuracy_score, precision_score

test=pd.read_csv('test.csv')
test_id=test['PassengerId']
file1=open('da_model.pkl','rb')

imputer1=pickle.load(file1)
imputer2=pickle.load(file1)
scaler=pickle.load(file1)
oencoder=pickle.load(file1)
sencoder=pickle.load(file1)
knnimputer=pickle.load(file1)
lr=pickle.load(file1)


#MODEL IMPLEMENTATION
#PROCESSING
test=pd.read_csv('test.csv')
def split(df):
    df['Deck']=df['Cabin'].str.split('/',expand=True)[0]
    df['Num']=df['Cabin'].str.split('/',expand=True)[1]
    df['Side']=df['Cabin'].str.split('/',expand=True)[2]
    df=df.drop(columns=['Cabin'])
    
    df['GroupId']=df['PassengerId'].str.split('_',expand=True)[0]
    df['MemberId']=df['PassengerId'].str.split('_',expand=True)[1]
    df=df.drop(columns=['PassengerId'])
    return df
def process_test(df):
    df=split(df)
    df[cat_cols] = imputer1.transform(df[cat_cols])
    df['Num'] = knnimputer.transform(df[['Num']])
    df['Age'] = imputer2.transform(df[['Age']])
    df[spend_cols]=df[spend_cols].fillna(0)
    df[bin_cols]=df[bin_cols].fillna(False).astype(str)
    
    encoded_vals = sencoder.transform(df[cat_cols])
    encoded_df = pd.DataFrame(encoded_vals, columns=sencoder.get_feature_names_out(cat_cols), index=df.index)
    df = pd.concat([df.drop(columns=cat_cols), encoded_df], axis=1)
    
    for col in bin_cols:
        df[col] = df[col].map({'True': 1, 'False': 0})
        
    df[num_cols] = scaler.transform(df[num_cols])
    df.drop(columns=dropped,inplace=True)
 
    return df
#ANALYSIS
test_pro=process_test(test.copy())
predicted=lr.predict(test_pro)
predicted_bo=predicted.astype(bool)
submission=pd.DataFrame({'PassengerId':test_id,'Transported':predicted_bo})
print('The final result is', submission.to_csv)