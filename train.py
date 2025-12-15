<<<<<<< Updated upstream
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# LOAD DATA 
train = pd.read_csv("train.csv")


# FEATURE ENGINEERING
# Split Name
train["FirstName"] = train["Name"].str.split(" ").str[0]
train["LastName"] = train["Name"].str.split(" ").str[1]
train.drop(columns = "Name", inplace = True)

# Split Cabin
train["Deck"] = train["Cabin"].str.split("/").str[0]
train["Num"] = train["Cabin"].str.split("/").str[1]
train["Side"] = train["Cabin"].str.split("/").str[2]
train.drop(columns = "Cabin", inplace = True)

# Split PassengerId
train["GroupId"] = train["PassengerId"].str.split("_").str[0]
train["MemberNum"] = train["PassengerId"].str.split("_").str[1].astype("int")
train.drop(columns = "PassengerId", inplace = True)

# Group Size
groupSizeMap = train["GroupId"].value_counts().to_dict()
train["GroupSize"] = train["GroupId"].map(groupSizeMap)

# # Family Size
# familySizeMap = train["LastName"].value_counts().to_dict()
# train['FamilySize'] = train["LastName"].map(familySizeMap)

# Boolean Mapping
booleanMap = {True: 1, False: 0}
train["Transported"] = train["Transported"].map(booleanMap)
train["CryoSleep"] = train["CryoSleep"].map(booleanMap)
train["VIP"] = train["VIP"].map(booleanMap)


# IMPUTATION
# NOTE : Initial Temporary Imputation

# Fill Text or Categorical
colsText = ['HomePlanet', 'Destination', 'Deck', 'Side', 'FirstName', 'LastName', 'Num']
for col in colsText:
    if col in train.columns:
        train[col] = train[col].fillna(train[col].mode()[0])

# Fill Boolean
colsBool = ['CryoSleep', 'VIP']
for col in colsBool:
    train[col] = train[col].fillna(0)

# Fill Numerical
if 'Age' in train.columns:
    train['Age'] = train['Age'].fillna(train['Age'].median())

# Fill Spending
spendCols = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
train[spendCols] = train[spendCols].fillna(0)


# SCALING
# NOTE : Initial Temporary Scaling

# Spend Columns
spendCols = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
train[spendCols] = train[spendCols] / train[spendCols].max().max()

# Age and GroupSize
scaler = StandardScaler()
cols = ['Age', 'GroupSize']
if all(c in train.columns for c in cols):
    train[cols] = scaler.fit_transform(train[cols])


# PCA
# Run PCA on Spend Columns
pca = PCA(n_components = 2)
pcaData = pca.fit_transform(train[spendCols])
train['spendPC1'] = pcaData[:, 0]
train['spendPC2'] = pcaData[:, 1]
train.drop(columns = spendCols, inplace = True)


# ENCODING
# One Hot Encoding on HomePlanet, Destination, Deck, Side
oheCols = ['HomePlanet', 'Destination', 'Deck', 'Side']
ohe = OneHotEncoder(drop = 'first', sparse_output = False)
encodedData = ohe.fit_transform(train[oheCols])
feature_names = ohe.get_feature_names_out(oheCols)
encodedDF = pd.DataFrame(encodedData, columns = feature_names, index = train.index)
train = pd.concat([train.drop(columns = oheCols), encodedDF], axis = 1)


# CLEANUP AND SCALE
dropCols = ["FirstName", "LastName", "Num", "GroupId", "MemberNum"]
train.drop(columns = dropCols, inplace = True)


# MODEL TRAINING
x = train.drop('Transported', axis=1)
y = train['Transported']

xTrain, xVal, yTrain, yVal = train_test_split(x, y, test_size = 0.2, random_state = 0)

model = LogisticRegression(max_iter = 10000)
model.fit(xTrain, yTrain)

preds = model.predict(xVal)
accuracy = accuracy_score(yVal, preds)

print(f"Final Validation Accuracy: {accuracy:.4f}")
print("Model Trained Successfully")

# print(train.sample(10, random_state = 0))
# print(train.columns)
=======
#TRAINING
#Preprocessing Model

import sklearn.preprocessing as sp
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import sklearn.impute as si

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sb
import joblib

from sklearn.metrics import accuracy_score, precision_score


df = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
test_id = test['PassengerId']


spend_cols = ['Spa', 'FoodCourt', 'RoomService', 'VRDeck', 'ShoppingMall']
num_cols = ['Age', 'Num'] + spend_cols
cat_cols = ['HomePlanet', 'Destination', 'Deck', 'Side']
bin_cols = ['CryoSleep', 'VIP']
dropped = ['Name', 'GroupId', 'MemberId']
final = ['Transported']


sencoder = sp.OneHotEncoder(sparse_output=False, handle_unknown='ignore')
scaler = sp.StandardScaler()
imputer1 = si.SimpleImputer(strategy='most_frequent')
imputer2 = si.SimpleImputer(strategy='median')
knnimputer = si.KNNImputer(n_neighbors=5)
oencoder = sp.OrdinalEncoder()

def split(df):
    df['Deck'] = df['Cabin'].str.split('/', expand=True)[0]
    df['Num'] = df['Cabin'].str.split('/', expand=True)[1]
    df['Side'] = df['Cabin'].str.split('/', expand=True)[2]
    df = df.drop(columns = ['Cabin'])
    
    df['GroupId'] = df['PassengerId'].str.split('_', expand=True)[0]
    df['MemberId'] = df['PassengerId'].str.split('_', expand=True)[1]
    df = df.drop(columns = ['PassengerId'])
    return df

def preprocessing(df):
    df = split(df)
    df['Num'] = pd.to_numeric(df['Num'], errors='coerce')
    df['Num'] = df['Num'].astype('Int64')
    
    def impute(df):
        df[cat_cols] = imputer1.fit_transform(df[cat_cols])
        df['Num'] = knnimputer.fit_transform(df[['Num']])
        df['Age'] = imputer2.fit_transform(df[['Age']])
        num_cols = ['Age', 'Num'] + spend_cols
        df[spend_cols] = df[spend_cols].fillna(0)
        df[bin_cols] = df[bin_cols].fillna(False)
        return df
    
    df = impute(df)

    def encode(df):
        encoded_vals = sencoder.fit_transform(df[cat_cols])
        encoded_cols = sencoder.get_feature_names_out(cat_cols)
        encoded_df = pd.DataFrame(encoded_vals, columns=encoded_cols, index=df.index)
        df = pd.concat([df.drop(columns=cat_cols), encoded_df], axis=1)
        logic = {True: 1, False: 0}
        for col in bin_cols:
            df[col] = df[col].map(logic)
        return df
    
    df = encode(df)

    def scale(df):
        df[num_cols] = scaler.fit_transform(df[num_cols])
        return df
    scale(df)

    def drop(df):
        df.drop(columns=dropped, inplace=True)
        return df
    drop(df)

    return df


# Analysis Model
df = preprocessing(df)
x = df.drop(['Transported'], axis=1)
y = df['Transported']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
lr = LogisticRegression(max_iter=2000)
lr.fit(x_train,y_train)
y_pred = lr.predict(x_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Final Validation Accuracy: {accuracy:.4f}")
print("Model Trained Successfully")

bundle = {
    'imputer1': imputer1,
    'imputer2': imputer2,
    'scaler': scaler,
    'oencoder': oencoder,
    'sencoder': sencoder,
    'knnimputer': knnimputer,
    'lr': lr
}

joblib.dump(bundle, 'da_model.pkl')
>>>>>>> Stashed changes
