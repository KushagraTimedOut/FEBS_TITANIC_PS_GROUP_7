import sklearn.preprocessing as sp
import sklearn.impute as si
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sb
import joblib

from sklearn.metrics import accuracy_score, precision_score

test = pd.read_csv('test.csv')
test_id = test['PassengerId']

bundle = joblib.load('da_model.pkl')

imputer1 = bundle['imputer1']
imputer2 = bundle['imputer2']
scaler = bundle['scaler']
oencoder = bundle['oencoder']
sencoder = bundle['sencoder']
knnimputer = bundle['knnimputer']
lr = bundle['lr']

spend_cols = ['Spa', 'FoodCourt', 'RoomService', 'VRDeck', 'ShoppingMall']
num_cols = ['Age', 'Num'] + spend_cols
cat_cols = ['HomePlanet', 'Destination', 'Deck', 'Side']
bin_cols = ['CryoSleep', 'VIP']
dropped = ['Name', 'GroupId', 'MemberId']
final = ['Transported']

# MODEL IMPLEMENTATION
# PROCESSING

def split(df):
    df['Deck'] = df['Cabin'].str.split('/', expand=True)[0]
    df['Num'] = df['Cabin'].str.split('/', expand=True)[1]
    df['Side'] = df['Cabin'].str.split('/', expand=True)[2]
    df = df.drop(columns = ['Cabin'])
    
    df['GroupId'] = df['PassengerId'].str.split('_', expand=True)[0]
    df['MemberId'] = df['PassengerId'].str.split('_', expand=True)[1]
    df = df.drop(columns = ['PassengerId'])
    return df

def process_test(df):
    df = split(df)
    df[cat_cols] = imputer1.transform(df[cat_cols])
    df['Num'] = knnimputer.transform(df[['Num']])
    df['Age'] = imputer2.transform(df[['Age']])
    df[spend_cols] = df[spend_cols].fillna(0)
    df[bin_cols] = df[bin_cols].fillna(False)
    
    encoded_vals = sencoder.transform(df[cat_cols])
    encoded_df = pd.DataFrame(encoded_vals, columns=sencoder.get_feature_names_out(cat_cols), index=df.index)
    df = pd.concat([df.drop(columns=cat_cols), encoded_df], axis=1)
    
    for col in bin_cols:
        df[col] = df[col].map({True: 1, False: 0})
        
    df[num_cols] = scaler.transform(df[num_cols])
    df.drop(columns=dropped,inplace=True)
 
    return df

#ANALYSIS
test_pro = process_test(test.copy())
predicted = lr.predict(test_pro)
predicted_bo = predicted.astype(bool)
submission = pd.DataFrame({'PassengerId': test_id, 'Transported': predicted_bo})
print('The final result is\n', submission)
submission.to_csv("submission.csv", index=False)