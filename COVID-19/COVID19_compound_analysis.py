#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Cheminformatics analysis of covid19 compounds

The separate functions should be integrated into specific development pipelines

"""
#%% Libraries-----------------------------------------------------
import rdkit
from rdkit import DataStructs
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import PandasTools
from rdkit.Chem import Descriptors

from rdkit.Chem import Draw
from rdkit.Chem.Draw import IPythonConsole


import numpy as np
import pandas as pd 
import seaborn as sns
from collections import Counter
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from rdkit.Chem.Draw import SimilarityMaps
from IPython.display import SVG
IPythonConsole.ipython_useSVG=True

from rdkit.Chem import ReplaceCore, GetMolFrags, ReplaceSubstructs, CombineMols
from itertools import product
from rdkit.Chem import rdFMCS

#R groups decomposition
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem import rdRGroupDecomposition
from rdkit.Chem import rdDepictor
from rdkit.Chem import rdmolops
from rdkit.Chem import RDConfig
from rdkit.Chem.Scaffolds import MurckoScaffold

#data base search
import pubchempy as pcp
from chembl_webresource_client.new_client import new_client

from tqdm.auto import tqdm

import re
import random
import itertools
import os
#os.getcwd()
os.chdir('D:/Research/covid19_research/COVID19')
#%% Load data -----------------------------------------------------

#sdf=PandasTools.LoadSDF("PubChem_compound_text_covid-19 clinicaltrials_records.sdf", smilesName="SMILES", molColName="Molecule")

data=pd.read_csv("Rdkit_OT_data_with_cluster_list.csv")

#visualise the loaded cpd library
#data.info()
data.head()

#%% -----------------------------------------------------
#convert to molecules and drop unecessary columns from the dataframe

#sdf=sdf.drop(["PUBCHEM_IUPAC_OPENEYE_NAME","PUBCHEM_IUPAC_CAS_NAME","PUBCHEM_IUPAC_NAME_MARKUP","PUBCHEM_OPENEYE_CAN_SMILES","PUBCHEM_OPENEYE_ISO_SMILES"],axis=1)

data["RMol"]=[Chem.MolFromSmiles(_) for _ in data["canonicalSmiles"]]
data.head()

#%% -----------------------------------------------------
#Draw molecules
Draw.MolsToGridImage(data.RMol[1:10],molsPerRow=4, legends=list(data.drugName[1:10]))
#%% Pre-process data -----------------------------------------------------

#check if there are no duplicates

data['canonicalSmiles'].nunique() #158
data['canonicalSmiles'].count() #158 one duplicate value

#data=data.drop_duplicates(['canonicalSmiles'])
#%%-----------------------------------------------------
#Characterise molecules

#Add H for the analysis (create a separate RMol column)
data["RMol_H"]=[Chem.MolFromSmiles(_) for _ in data["canonicalSmiles"]]
data.RMol_H=data.RMol_H.apply(lambda x: Chem.AddHs(x))
data.head()

#Draw molecules
#Draw.MolsToGridImage(data.RMol_H[1:10],molsPerRow=4, legends=list(data.drugName[1:10]))

#%%
#Add descriptors
#Some descriptors treat H explicitly other implicitly

data["Atom_number"]=data.RMol_H.apply(lambda x: x.GetNumAtoms())
data["MW"]=data.RMol_H.apply(lambda x: Descriptors.MolWt(x))
data["MolLogP"]=data.RMol_H.apply(lambda x: Descriptors.MolLogP(x))


data["TSPA"]=data.RMol_H.apply(lambda x: Chem.rdMolDescriptors.CalcTPSA(x))
data["HBD_count"]=data.RMol_H.apply(lambda x: Chem.rdMolDescriptors.CalcNumHBD(x))
data["HBA_count"]=data.RMol_H.apply(lambda x: Chem.rdMolDescriptors.CalcNumHBA(x))
data["Rotatable_bond_count"]=data.RMol_H.apply(lambda x:Descriptors.NumRotatableBonds(x))
data["Aromatic_atom_count"]=data.RMol_H.apply(lambda x: sum([ x.GetAtomWithIdx(i).GetIsAromatic() for i in range(x.GetNumAtoms())]))

data["Ring_number"]=data.RMol_H.apply(lambda x: Chem.rdMolDescriptors.CalcNumRings(x))
data["Number_heterocycles"]=data.RMol_H.apply(lambda x: Chem.Lipinski.NumAromaticRings(x))
data["Heavy_atom_number"]=data.RMol_H.apply(lambda x: Descriptors.HeavyAtomCount(x))
 
#%%-----------------------------------------------------
#Sometimes substruct matches are better to differentiate various attoms Cl and C


#data["S_count"]=data.canonicalSmiles.apply(lambda x:  Counter(x)["S"] if "S" in Counter(x).keys() else 0 )

data["C_count"]=data.RMol.apply(lambda x: len(list(itertools.chain(*(x.GetSubstructMatches(Chem.MolFromSmiles("C")))))))
data["F_count"]=data.RMol.apply(lambda x: len(list(itertools.chain(*(x.GetSubstructMatches(Chem.MolFromSmiles("F")))))))
data["O_count"]=data.RMol.apply(lambda x:  len(list(itertools.chain(*(x.GetSubstructMatches(Chem.MolFromSmiles("O")))))))
data["N_count"]=data.RMol.apply(lambda x:  len(list(itertools.chain(*(x.GetSubstructMatches(Chem.MolFromSmiles("N")))))))
data["Cl_count"]=data.RMol.apply(lambda x: len(list(itertools.chain(*(x.GetSubstructMatches(Chem.MolFromSmiles("Cl")))))))
data["P_count"]=data.RMol.apply(lambda x:  len(list(itertools.chain(*(x.GetSubstructMatches(Chem.MolFromSmiles("P")))))))
data["S_count"]=data.RMol.apply(lambda x:  len(list(itertools.chain(*(x.GetSubstructMatches(Chem.MolFromSmiles("S")))))))


data["AP"]=data.apply(lambda row: row.Aromatic_atom_count/row.Heavy_atom_number if row.Heavy_atom_number>0 else 0, axis=1)

#%%-----------------------------------------------------

sns.pairplot(data[["AP","Ring_number","TSPA","MW","MolLogP","Atom_number","C_count","Number_heterocycles"]], diag_kind="kde",kind="reg",markers="+")
plt.show()

#%%------------------------------------------------------

sns.pairplot(data[["AP","C_count","N_count","MW","MolLogP","O_count","P_count","Atom_number"]], diag_kind="kde",kind="reg",markers="+")
plt.show()

#%% Add Morgan fingerprints-----------------------------------------------------

ECFP4=[AllChem.GetMorganFingerprintAsBitVect(_, radius=2, nBits=2048) for _ in data["RMol"]]
data["Morgan_fp"]=ECFP4

#convert RDKit explicit vectors into numpy arrays

ECFP4_np=[]

for fp in ECFP4:
    arr=np.zeros((1,))
    DataStructs.ConvertToNumpyArray(fp,arr)
    ECFP4_np.append(arr)

#%% Similarity calculation for all compounds-----------------------------------------------------

#Build array wit similary scores

length=len(ECFP4)

array=pd.DataFrame(index=range(length), columns=range(length))
array.columns=list(data.drugName)
array.index=list(data.drugName)

for i in range(length):
    mol1=list(data.drugName)[i]
    fp1=list(data.Morgan_fp)[i]
    for j in range(length):
        mol2=list(data.drugName)[j]
        fp2=list(data.Morgan_fp)[j]
        #calculate similarity
        similarity=DataStructs.FingerprintSimilarity(fp1, fp2)
        array.iloc[i,j]=similarity


#%% Compound assessment-----------------------------------------------------


    color_dict={"Cluster 1": 'tab:cyan',"Cluster 2": 'tab:orange',"Cluster 3": 'tab:green',"Cluster 4": 'tab:blue',"Cluster 5": 'tab:red', "Not assigned": 'tab:grey'}
    color_array=[color_dict[var] for var in data.Cluster_list]
    color_df=pd.DataFrame({"Drug Name": data.drugName, "Color": color_array})
    array=array.astype(float)

    #array.to_csv("cluster_matrix.csv")
    #plot heatmap and dendogram
   
    sns.clustermap(array, metric="euclidean", method="single", figsize=(30, 30), xticklabels=True ,yticklabels=True, row_colors=color_array)
    handles = [Patch(facecolor=color_dict[name]) for name in color_dict]
    plt.legend(handles, color_dict, title='Clusters', bbox_to_anchor=(1, 1), bbox_transform=plt.gcf().transFigure, loc='upper right')
    plt.show()




#%%  -----------------------------------------------------
#Explore MCS for compounds

clust_5=data[data.Cluster_list=="Cluster 5"]


#Use Murcko scaffolds to refine search
scaffold=[MurckoScaffold.GetScaffoldForMol(mol) for mol in clust_5.RMol]

#either molecules or list(clust_2.RMol) or Murcko scaffolds can be checked for MCS
MCS_scaffold=rdFMCS.FindMCS(scaffold, threshold=0.5,completeRingsOnly=True,matchValences=True, bondCompare=rdFMCS.BondCompare.CompareOrderExact, atomCompare=rdFMCS.AtomCompare.CompareElements) #completeRingsOnly=True, matchValences=True,bondCompare=rdFMCS.BondCompare.CompareOrderExact, atomCompare=rdFMCS.AtomCompare.CompareElements

Chem.MolFromSmarts(MCS_scaffold.smartsString)

#%%-------------------------------------------------------
#Build cluster 5 array

#Build array wit similary scores

cpd_names=list(clust_5.drugName)

#subset array
#subset_array=array.loc[cpd_names,cpd_names]
select_ids=[ drug in cpd_names for drug in data.drugName]
subset_array=data[select_ids]

# build similarity array
length=len(cpd_names)
clust5_similarity_array=pd.DataFrame(index=range(length), columns=range(length))
clust5_similarity_array.columns=list(subset_array.drugName)
clust5_similarity_array.index=list(subset_array.drugName)

for i in range(length):
    mol1=list(data.drugName)[i]
    fp1=list(data.Morgan_fp)[i]
    for j in range(length):
        mol2=list(data.drugName)[j]
        fp2=list(data.Morgan_fp)[j]
        #calculate similarity
        similarity=DataStructs.FingerprintSimilarity(fp1, fp2)
        clust5_similarity_array.iloc[i,j]=round(similarity,2)

#clust5_similarity_array.to_csv("clust5_similarity.csv")
#clust5_similarity_array>0.5
#clust5_similarity_array[np.where(clust5_similarity_array>0.2)]

#%%  -----------------------------------------------------
#Peform similarity analysis for cluster 5

cpd_names=list(clust_5.drugName)

#subset array
#subset_array=array.loc[cpd_names,cpd_names]
select_ids=[ drug in cpd_names for drug in data.drugName]
subset_array=data[select_ids]

# build similarity array
length=len(cpd_names)
similarity_array=pd.DataFrame(index=range(length), columns=range(length))
similarity_array.columns=list(subset_array.drugName)
similarity_array.index=list(subset_array.drugName)

for m1 in similarity_array.index:
    for m2 in similarity_array.columns:
        if m1==m2:
                similarity_array.loc[m1,m2]=list(subset_array["RMol"])[list(subset_array.drugName).index(m1)]
                continue
        #m2 is drawn so the array should automatically adjust with upper and lower parts
        #first molecule is ref molecule and the second is the target molecule
        
        m1_struct=list(subset_array["RMol"])[list(subset_array.drugName).index(m1)]
        m2_struct=list(subset_array["RMol"])[list(subset_array.drugName).index(m2)]
        
        fig, maxweight = SimilarityMaps.GetSimilarityMapForFingerprint(m1_struct, m2_struct, lambda m,idx: SimilarityMaps.GetMorganFingerprint(m, atomId=idx, radius=2, fpType='count'), metric=DataStructs.TanimotoSimilarity)
        similarity_array.loc[m1,m2]=fig

#figures are stored in similarity_array
#images=list(similarity_array.values.reshape(-1))

#First molecule is reference the other is the target
#similarity_array.loc["Dexmedetomidine","Melatonin"]
#similarity_array.loc["Melatonin","Dexmedetomidine"]
#similarity_array.loc["Prasugrel","Ramelteon"]
#similarity_array.loc["Ramelteon","Prasugrel"]
#similarity_array.loc["Prasugrel","Cenicriviroc"]
#similarity_array.loc["Cenicriviroc","Prasugrel"]

#Draw.MolsToImage(data.loc[data.drugName=="Prasugrel",]["RMol"])

#%%  -----------------------------------------------------
#PubChem download

cid_max = 138962044    # The maximum CID in PubChem as of September 2019

cids = []

random.seed(0)
for x in range(20000):
    cids.append(random.randint(1, cid_max + 1))

chunks=round(len(cids)/100) #separate data into chunks for downloading

cmpd_list=[]
for chunk in tqdm(np.array_split(cids, chunks)):

    cmpd_list.append(pcp.get_compounds(chunk.tolist()))



#%%  -----------------------------------------------------

cmpd_list=list(itertools.chain(*cmpd_list))
#Transform compound data

sample_df=pd.DataFrame(cids)
sample_df.rename(columns={0:"CID"}, inplace=True)

sample_df["Molecule"]=cmpd_list
sample_df["SMILES"]=[x.canonical_smiles  for x in cmpd_list]
sample_df["RMol"]=sample_df.SMILES.apply(lambda x: Chem.MolFromSmiles(x))

#Check for missing molecules
sum(sample_df["RMol"].isna())
sample_df.dropna(subset=["RMol"],inplace=True)
sample_df.shape
#Calculate morgan fingerprint for the sample

ECFP4=[AllChem.GetMorganFingerprintAsBitVect(_, radius=2, nBits=2048) for _ in sample_df["RMol"]]
sample_df["Morgan_fp"]=ECFP4

#%%  -----------------------------------------------------
#Perform similarity search for a selected cluster

clust_5=data[data.Cluster_list=="Cluster 5"]

threshold=0.4
matches={}

for idx1 in list(clust_5.index):
    cpd1=clust_5.drugName[idx1]
    fp1=clust_5.Morgan_fp[idx1]

    matches[cpd1]=[]
    for idx2 in list(sample_df.index):

        cpd2=sample_df["Molecule"][idx2]
        fp2=sample_df["Morgan_fp"][idx2]


        similarity=DataStructs.FingerprintSimilarity(fp1, fp2)

        if(similarity>=threshold):
            matches[cpd1]+=[cpd2]

            
#%%  -----------------------------------------------------
#Find compounds similar to given SMILES query with similarity threshold    
#CHEMBL database

#Search against cluster 5

threshold=40
matches_chembl={}

similarity = new_client.similarity

for idx1 in list(clust_5.index):
    cpd1=clust_5.drugName[idx1]
    smiles1=clust_5.canonicalSmiles[idx1]

    #Tanimoto similarity
    res = similarity.filter(smiles=smiles1, similarity=threshold).only(['molecule_chembl_id', 'similarity','pref_name'])

    matches_chembl[cpd1]=[]

    for cpd in res:
        #modification to ensure that drug names are returned
        if cpd['pref_name']==None:
            continue
        #ensure that different formulations are not matched
        if  re.search(cpd1,cpd['pref_name'], re.IGNORECASE):
            continue
            
        matches_chembl[cpd1]+=[cpd]

#%%------------------------------------------------------
#Reformat into a data frame
matches_df=pd.DataFrame(columns=['molecule_chembl_id', 'pref_name', 'similarity', 'Drug'])

for name in matches_chembl.keys():

    if len(matches_chembl[name])==0:
        continue

    temp_df=pd.DataFrame(matches_chembl[name])
    temp_df['Drug']=[name for i in range(temp_df.shape[0])]
    matches_df=pd.concat([matches_df,temp_df])


 
#%%  -----------------------------------------------------
# Check matching drugs  and save filtered table
list1=list(matches_df.pref_name.apply(lambda x: x.title()))
list2=list(data.drugName)


shared_drugs=[drug in list2 for drug in list1] #Matches
matches_df[shared_drugs] #PRASUGREL, CLOPIDOGREL, TXA127
sum(shared_drugs) #3 matches
len(matches_df.molecule_chembl_id.unique()) #490

#unique

unique_matches_df=matches_df[~matches_df.pref_name.isin(["PRASUGREL", "CLOPIDOGREL", "TXA127"])]
len(unique_matches_df.molecule_chembl_id.unique()) #487
#unique_matches_df.to_csv("CHEMBL_data_minimg_Suppl_Table3.csv")


