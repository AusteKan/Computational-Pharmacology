# -*- coding: utf-8 -*-
"""COVID19_DL.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1Ax4s5gy8_iamhhis_f2Xld1qZLIFL9RI
"""

!pip install rdkit-pypi

#Installing a package
!pip install git+https://github.com/samoturk/mol2vec;

#Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import re
import random
import itertools
from collections import OrderedDict
from itertools import chain

from rdkit import DataStructs
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import PandasTools
from rdkit.Chem import Descriptors
from rdkit.Chem import Draw

#machine learning 
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error,r2_score, accuracy_score,plot_confusion_matrix,roc_auc_score,classification_report,recall_score, precision_score
from sklearn.model_selection import train_test_split

#Deep learning
import tensorflow as tf

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import LSTM


from mol2vec.features import mol2alt_sentence, mol2sentence, MolSentence, DfVec, sentences2vec
from gensim.models import word2vec

from google.colab import files
uploaded = files.upload()

#Load data

antivirals=pd.read_csv("Antiviral_collection.csv") #active selection
chembl_db=pd.read_csv("chembl_db_selected.csv") #inactive selection
#data=pd.read_csv("Rdkit_OT_data_with_cluster_list.csv") #RDKIT data for clincal trial compounds
Mpro=pd.read_csv("Mpro.csv")

print(antivirals.shape) #48876
print(chembl_db.shape) #50000
#print(data.shape) #158
print(Mpro.shape) #95

from gensim.models import word2vec
model_m2v = word2vec.Word2Vec.load('model_300dim.pkl')

#Calculate morgan fingerprint for the active antiviral sample
antivirals["RMol"]=antivirals.SMILES.apply(lambda x: Chem.MolFromSmiles(x))
#check for missing values
sum(antivirals["RMol"].isna())
antivirals.dropna(subset=["RMol"],inplace=True)

#Calculate morgan fingerprint for the Chembl dataset

chembl_db["Smiles"]=chembl_db["Smiles"].astype(str)
chembl_db["RMol"]=[Chem.MolFromSmiles(_) for _ in chembl_db["Smiles"]]
#check for missing values
sum(chembl_db["RMol"].isna())
chembl_db.dropna(subset=["RMol"],inplace=True)
chembl_db.reset_index(drop=True, inplace=True)

#Calculate morgan fingerprint for the Mpro dataset

Mpro["SMILES"]=Mpro["SMILES"].astype(str)
Mpro["RMol"]=[Chem.MolFromSmiles(_) for _ in Mpro["SMILES"]]
#check for missing values
sum(Mpro["RMol"].isna())
Mpro.dropna(subset=["RMol"],inplace=True)
Mpro.reset_index(drop=True, inplace=True)

#Add type info

antivirals['Type']=["Active" for _ in range(len(antivirals.SMILES))]
chembl_db['Type']=["Inactive" for _ in range(len(chembl_db.Smiles))]

#Rename columns
antivirals=antivirals[['ID', 'SMILES', 'Type','RMol']]

chembl_db=chembl_db[['ChEMBL ID', 'Type', 'Smiles', 'RMol']]
chembl_db.rename(columns={'ChEMBL ID':'ID', 'Type':'Type', 'Smiles':'SMILES', 'RMol':'RMol'}, inplace=True)

data_df=pd.concat([antivirals,chembl_db])

print(data_df.shape) #98732, 3
data_df.head()

#Constructing sentences
data_df['Sentence'] = data_df.apply(lambda x: MolSentence(mol2alt_sentence(x['RMol'], 1)), axis=1)

#Extract embeddings to a np.array
#Note to always mark unseen='UNK' in sentence2vec() so that model can be taught how to handle unknown substructures
data_df['mol2vec'] = [DfVec(x) for x in sentences2vec(data_df['Sentence'], model_m2v, unseen='UNK')]
X = np.array([x.vec for x in data_df['mol2vec']])

Mpro.head()

#Constructing sentences for MPro
Mpro['Sentence'] = Mpro.apply(lambda x: MolSentence(mol2alt_sentence(x['RMol'], 1)), axis=1)
#Extract embeddings to a np.array
#Note to always mark unseen='UNK' in sentence2vec() so that model can be taught how to handle unknown substructures
Mpro['mol2vec'] = [DfVec(x) for x in sentences2vec(Mpro['Sentence'], model_m2v, unseen='UNK')]
X_Mpro = np.array([x.vec for x in Mpro['mol2vec']])

X_Mpro.shape #95, 300

X.shape #98732, 300

#Add labels for comppounds
labels = data_df['Type'].unique().tolist()

#Use label encoder to transform data
le = LabelEncoder()
le.fit(labels)
data_df['Label'] = le.transform(data_df["Type"])
y=data_df.Label

X_train,X_test,y_train, y_test=train_test_split(X,y, test_size=0.2, random_state=1,shuffle=True)

#Standardize data may not be needed as mol2vec already transformed the data
#std_scaler=StandardScaler()
#X_train=std_scaler.fit_transform(X_train)
#X_test=std_scaler.transform(X_test)

tf.keras.backend.clear_session()

# Create the model 
from keras import layers
from keras.layers.embeddings import Embedding
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D


model = Sequential()
model.add(layers.Dense(200,input_shape=(300,),activation='relu'))
model.add(Dropout(0.25))
model.add(layers.Dense(128,activation='relu'))
model.add(Dropout(0.25))
model.add(layers.Dense(100,activation='relu'))
model.add(Dropout(0.25))
model.add(layers.Dense(50,activation='relu'))
model.add(Dropout(0.25))
model.add(layers.Dense(1,activation='sigmoid'))

# Compile the model
model.compile(loss = 'binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit the model
history = model.fit(X_train, y_train, epochs = 200, batch_size = 256, validation_data=(X_test,y_test))

history.history.keys()

#Evaluate model
loss=history.history['loss']
val_loss=history.history['val_loss']

acc=history.history['accuracy']
val_acc=history.history['val_accuracy']
epochs=range(1,len(acc)+1)

plt.plot(epochs,loss,'bo',label="Training loss")
plt.plot(epochs,val_loss,'b',label="Validation loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

plt.plot(epochs,acc,'bo',label="Training accuracy")
plt.plot(epochs,val_acc,'b',label="Validation accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

#Evaluate model
results=model.evaluate(X_test,y_test)
results

#Predict the probability that reviews are positive
predicted=model.predict(X_test)

#Mpro evaluation
Mpro_predicted=model.predict(X_Mpro)

Mpro_list_predictions=np.concatenate(Mpro_predicted).ravel().tolist()

sum([val >0.5 for val in Mpro_list_predictions])

Mpro_list_predictions

#Test model with OT drug data set
data=pd.read_csv("Rdkit_OT_data_with_cluster_list.csv") #RDKIT data for clincal trial compounds

data.columns

#Calculate morgan fingerprint for the active antiviral sample
data["RMol"]=data.canonicalSmiles.apply(lambda x: Chem.MolFromSmiles(x))
#check for missing values
sum(data["RMol"].isna())
data.dropna(subset=["RMol"],inplace=True)
#Constructing sentences
data['Sentence'] = data.apply(lambda x: MolSentence(mol2alt_sentence(x['RMol'], 1)), axis=1)
#Extract embeddings to a np.array
#Note to always mark unseen='UNK' in sentence2vec() so that model can be taught how to handle unknown substructures
data['mol2vec'] = [DfVec(x) for x in sentences2vec(data['Sentence'], model_m2v, unseen='UNK')]
OT_X = np.array([x.vec for x in data['mol2vec']])

data.shape #158 10

OT_X.shape #158,300

#OT evaluation
OT_predicted=model.predict(OT_X)

OT_predicted_list=np.concatenate(OT_predicted).ravel().tolist()
sum([val >0.5 for val in OT_predicted_list])

actives=[val < 0.5 for val in OT_predicted_list] #actives are 0

print(data.drugName[actives])

Draw.MolsToGridImage(data.RMol[actives], molsPerRow=3,  legends=list(data.drugName[actives]))