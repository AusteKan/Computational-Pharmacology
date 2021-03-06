# -*- coding: utf-8 -*-
"""Covid19_antiviral_analysis.ipynb

The separate functions should be integrated into specific development pipelines
"""

!pip install rdkit-pypi

#Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import rdkit
from rdkit import DataStructs
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import PandasTools
from rdkit.Chem import Descriptors
from rdkit.Chem import Draw

import re
import random
import itertools


#machine learning 
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error,r2_score, accuracy_score,plot_confusion_matrix,roc_auc_score,classification_report,recall_score, precision_score

from google.colab import files
uploaded = files.upload()

#Load data

antivirals=pd.read_csv("Antiviral_collection.csv") #active selection
chembl_db=pd.read_csv("chembl_db_selected.csv") #inactive selection
data=pd.read_csv("Rdkit_OT_data_with_cluster_list.csv") #RDKIT data for clincal trial compounds
Mpro=pd.read_csv("Mpro.csv")

print(antivirals.shape) #48876
print(chembl_db.shape) #50000
print(data.shape) #158
print(Mpro.shape) #95

#Calculate morgan fingerprint for the active antiviral sample
antivirals["RMol"]=antivirals.SMILES.apply(lambda x: Chem.MolFromSmiles(x))
#check for missing values
sum(antivirals["RMol"].isna())
antivirals.dropna(subset=["RMol"],inplace=True)
#calculate Morgan fingerprints
ECFP4=[AllChem.GetMorganFingerprintAsBitVect(_, radius=3, nBits=2048) for _ in antivirals["RMol"]]
antivirals["Morgan_fp"]=ECFP4
antivirals.head()

#Calculate morgan fingerprint for the Chembl dataset

chembl_db["Smiles"]=chembl_db["Smiles"].astype(str)
chembl_db["RMol"]=[Chem.MolFromSmiles(_) for _ in chembl_db["Smiles"]]
#check for missing values
sum(chembl_db["RMol"].isna())
chembl_db.dropna(subset=["RMol"],inplace=True)
chembl_db.reset_index(drop=True, inplace=True)
#calculate Morgan fingerprints
ECFP4=[AllChem.GetMorganFingerprintAsBitVect(_, radius=3, nBits=2048) for _ in chembl_db["RMol"]]
chembl_db["Morgan_fp"]=ECFP4
chembl_db.head()

data.columns

#Calculate morgan fingerprint for the data dataset

data["canonicalSmiles"]=data["canonicalSmiles"].astype(str)
data["RMol"]=[Chem.MolFromSmiles(_) for _ in data["canonicalSmiles"]]
#check for missing values
sum(data["RMol"].isna())
data.dropna(subset=["RMol"],inplace=True)
data.reset_index(drop=True, inplace=True)
#calculate Morgan fingerprints
ECFP4=[AllChem.GetMorganFingerprintAsBitVect(_, radius=3, nBits=2048) for _ in data["RMol"]]
data["Morgan_fp"]=ECFP4
print(data.shape)
data.head()

#Calculate morgan fingerprint for the Mpro dataset

Mpro["SMILES"]=Mpro["SMILES"].astype(str)
Mpro["RMol"]=[Chem.MolFromSmiles(_) for _ in Mpro["SMILES"]]
#check for missing values
sum(Mpro["RMol"].isna())
Mpro.dropna(subset=["RMol"],inplace=True)
Mpro.reset_index(drop=True, inplace=True)
#calculate Morgan fingerprints
ECFP4=[AllChem.GetMorganFingerprintAsBitVect(_, radius=3, nBits=2048) for _ in Mpro["RMol"]]
Mpro["Morgan_fp"]=ECFP4
print(Mpro.shape)
Mpro.head()

#Lower similarity score
threshold=0.2

#NOTE you may use samples from active and inactive to reduce testing space and only use approximate evaluation
random_indexes1=[random.randint(0,len(antivirals.SMILES)) for _ in range(0,1000)]
random_indexes2=[random.randint(0,len(chembl_db.Smiles)) for _ in range(0,1000)]
#Filtered PubChem least similar compound indexes
least_similar_cpds=[] #less or equal to threshold

for idx1 in random_indexes1:
    
    fp1=antivirals.Morgan_fp[idx1]

    temp_list=[]
    for idx2 in random_indexes2:

        
        fp2=chembl_db.Morgan_fp[idx2]


        similarity=DataStructs.FingerprintSimilarity(fp1, fp2)

        if similarity<=threshold:
            temp_list+=[idx1]

    if len(temp_list)>0:
        least_similar_cpds+=[idx1]

print(idx1,len(least_similar_cpds))

#Modify fingerprints into arrays
#Active
antivirals["Morgan_fp"]=antivirals.Morgan_fp.apply(lambda x: np.array(x))

#Inactive
chembl_db["Morgan_fp"]=chembl_db.Morgan_fp.apply(lambda x: np.array(x))

#OT data set for covid19 clinical trials
data["Morgan_fp"]=data.Morgan_fp.apply(lambda x: np.array(x))

#Mpro data set 
Mpro["Morgan_fp"]=Mpro.Morgan_fp.apply(lambda x: np.array(x))

#Combine data frames

#Add type info

antivirals['Type']=["Active" for _ in range(len(antivirals.SMILES))]
chembl_db['Type']=["Inactive" for _ in range(len(chembl_db.Smiles))]

#Rename columns
antivirals=antivirals[['ID', 'SMILES', 'Type','RMol','Morgan_fp']]

chembl_db=chembl_db[['ChEMBL ID', 'Type', 'Smiles', 'RMol','Morgan_fp']]
chembl_db.rename(columns={'ChEMBL ID':'ID', 'Type':'Type', 'Smiles':'SMILES', 'RMol':'RMol','Morgan_fp':'Morgan_fp'}, inplace=True)

data_df=pd.concat([antivirals,chembl_db])

print(data_df.shape) #98732, 5
data_df.head()

#Add labels for comppounds
labels = data_df['Type'].unique().tolist()

#Use label encoder to transform data
le = LabelEncoder()
le.fit(labels)
data_df['Label'] = le.transform(data_df["Type"])

#Active compounds are labeled 0 and inactive 1
data_to_model=data_df["Morgan_fp"].values
data_label=data_df.Label

data_df.head()

data_df.tail()

data_df.shape

#Split data
X_train,X_test,y_train, y_test=train_test_split(data_to_model,data_label, test_size=0.2, random_state=1)

X_train.shape #78985

X_test.shape #19747

X_train[1:5]

#2048 or other vectors need to be stacked into single value
X_train=np.stack(X_train,axis=0)
X_test=np.stack(X_test,axis=0)

X_train.shape #78985, 2048

X_train[1:5,:]

#Standardize data
std_scaler=StandardScaler()
X_train_std=std_scaler.fit_transform(X_train)
X_test_std=std_scaler.transform(X_test)

X_train_std[1:5,:]

#Prepare model
lgbm=LGBMClassifier()
lgbm.fit(X_train_std,y_train)
#Model quick evaluation
predict=lgbm.predict(X_test_std)

#Evaluate model performance using the ROC AUC score. 

roc_auc_score(y_test,predict) #0.9677444794451111

#Evalute classification metrics
from sklearn.metrics import classification_report
print(classification_report(y_test, predict))

#Test for overfitting

#Model quick evaluation
predict_train=lgbm.predict(X_train_std)
roc_auc_score(y_train,predict_train) #0.971762340688462

import seaborn as sns

p_score=precision_score(y_test,predict,average='macro')

r_score=recall_score(y_test,predict,average='macro')
	
a_score=accuracy_score(y_test,predict)

sns.set_style("white")
sns.set_context('talk')
plt.rcParams["figure.figsize"] = (8,8)
plot_confusion_matrix(lgbm,X_test_std,y_test,display_labels=sorted(labels),cmap=plt.cm.Blues)
plt.title("Accuracy score: {} ".format(a_score))
plt.show()

from sklearn.metrics import roc_curve, auc, roc_auc_score
fpr, tpr, _ = roc_curve(y_test,predict)
roc_auc = roc_auc_score(y_test,predict)
roc_auc
plt.figure()
plt.plot(fpr, tpr)
plt.plot([0.0, 1.0], [0.0, 1.0], ls='--', lw=0.3, c='k')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic, AUC score: {}'.format(roc_auc))
plt.show()

#Standardize Mpro data to test
X_data=Mpro.Morgan_fp.values
X_data=np.stack(X_data,axis=0)
X_test_Mpro=std_scaler.transform(X_data)


#Model quick evaluation
predict_Mpro=lgbm.predict(X_test_Mpro)

#Assess how many of known antivirals are predicted to be antivirals
predict_Mpro #No active compounds

#Standardize data covid19 OT
X_data=data.Morgan_fp.values
X_data=np.stack(X_data,axis=0)
X_test_covid_OTstd=std_scaler.transform(X_data)


#Model quick evaluation
predict_OT=lgbm.predict(X_test_covid_OTstd)

sum(predict_OT)

#Get active compounds - label 0
values_select=[not bool(val) for val in predict_OT]
data.drugName[values_select]

Draw.MolsToGridImage(data.RMol[values_select],molsPerRow=4,legends=list(data.drugName[values_select]))

#use this for data selection without ML

name_list=["Epoprostenol","Decitabine", "Maraviroc","Hydroxychloroquine","Ritonavir", "Ribavirin", "Cobicistat",  "Baricitinib",   "Menthol", "Piclidenoson", "Etoposide",   "Eritoran",  "Amantadine"]
index=[name in name_list for name in list(data.drugName)]
predicted_active_OT=data[index]
predicted_active_OT.head()

#Evaluate how similar compounds are
predicted_active_OT=data[values_select]
predicted_active_OT.head()

predicted_active_OT.shape #13, 9

Draw.MolsToGridImage(predicted_active_OT.RMol,molsPerRow=4,legends=list(predicted_active_OT.drugName))

antivirals.head()

#Draw antivirals heatmap

matrix=np.zeros((antivirals.shape[0],antivirals.shape[0]))
#Compound names do not get assignments as compounds are coded

for idx1 in range(antivirals.shape[0]):
    
    fp1=antivirals.Morgan_fp[idx1]
    
    for idx2 in range(antivirals.shape[0]):

        fp2=antivirals.Morgan_fp[idx2]
        similarity=DataStructs.FingerprintSimilarity(fp1, fp2)
        matrix[idx1,idx2]=similarity

#plot heatmap and dendogram

sns.clustermap(matrix, metric="euclidean", method="single", figsize=(30, 30))
plt.title("Antiviral compound similarity map")
plt.legend()
plt.show()

#Compound names do not get assignments as compounds are coded

for idx1 in range(antivirals.shape[0]):
    
    fp1=antivirals.Morgan_fp[idx1]
    
    for idx2 in range(antivirals.shape[0]):

        fp2=antivirals.Morgan_fp[idx2]
        similarity=DataStructs.FingerprintSimilarity(fp1, fp2)
        matrix[idx1,idx2]=similarity

#plot heatmap and dendogram

sns.clustermap(matrix, metric="euclidean", method="single", figsize=(30, 30))
plt.title("Antiviral compound similarity map")
plt.legend()
plt.show()
