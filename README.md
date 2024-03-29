![Type](https://img.shields.io/badge/pharma%20-research-brightgreen)


# Computational-Pharmacology
Computational pharmacology research 

[Manuscript link](https://doi.org/10.1016/j.bpc.2022.106891)

# Computational Pharmacology: New Avenues for COVID-19 Therapeutics Search and Better Preparedness for Future Pandemic Crises
               
## Abstract

The COVID-19 pandemic created an unprecedented global healthcare emergency prompting the exploration of new therapeutic avenues, including drug repurposing. The large number of ongoing studies revealed pervasive issues in clinical research, such as the lack of accessible and organised data. Moreover, current shortcomings in clinical studies highlighted the need for a multi-faceted approach to tackle this health crisis. Thus, we set out to explore and develop new strategies for drug repositioning by employing computational pharmacology, data mining, systems biology, and computational chemistry to advance shared efforts in identifying key targets, affected networks, and potential pharmaceutical intervention options. Our study revealed that formulating pharmacological strategies should rely on both therapeutic targets and their networks. We showed how data mining can reveal regulatory patterns, capture novel targets, alert about side-effects, and help identify new therapeutic avenues. We also highlighted the importance of the miRNA regulatory layer and how this information could be used to monitor disease progression or devise treatment strategies. Importantly, our work bridged the interactome with the chemical compound space to better understand a complex COVID-19 drug space. Machine and deep learning allowed us to showcase limitations in current chemical libraries for COVID-19 suggesting that both in silico and experimental analyses should be combined to retrieve therapeutically valuable compounds. Based on the gathered data, we strongly advocate for taking this opportunity to establish robust practices for treating today’s and future infectious diseases by preparing solid analytical frameworks.

![Graphical abstract](https://github.com/AusteKan/Computational-Pharmacology/blob/main/COVID-19/Graphical_abstract.jpg)

## Methods

## Data collection and mining

Data for COVID-19 associated clinical trials and drugs involved in treatment and/or clinical investigation protocols were primarily retrieved from the [Open Targets](https://www.opentargets.org/) platform that curates information on clinical testing, known targets, and compound information. Mining (November, 2021) returned 1,375 target-drug pairs which constituted 230 unique drugs and 356 unique targets (i.e., some drugs have multiple main targets or different drug formulations). In addition, Open Targets were searched for compound and known target associations to extract the relevant chemical data (e.g., SMILES, InchiKey, etc.) - this provided information on 18,376 compounds. To expand and verify the data sets, the information was cross-referenced against PubChem COVID-19 records (1,625 compound data) (58)⁠ and the [STITCH database](http://stitch.embl.de/) containing compound-protein interaction data (15,473,939 interaction points). Additional interactome data was retrieved mining the [STRING database](https://string-db.org/) (135,660 interactions, 5922 new targets for the expanded interactome network)⁠. [Reactome database](https://reactome.org/) was used to extract information on relevant pathways⁠. miRNA database was used from the [OmicInt package](https://cran.r-project.org/web/packages/OmicInt/index.html) associated repository to mine non-conding interactions. [ChEMBL compound database](https://www.ebi.ac.uk/chembl/) (>2.1 M chemical entities) was used to search for similar and control compounds when investigating COVID-19 clinical trial drugs⁠. COVID-19 CAS and [Diamond/Xchem Mpro compound](https://www.diamond.ac.uk/covid-19/for-scientists/Main-protease-structure-and-XChem.html) repositories were used to extract predicted and experimentally tested antiviral drugs. 

## Computational pharmacology and bioinformatics analysis

COVID-19 clinical trial data mining, cleaning, and analysis was performed in R programming environment (v4.1.2) with RStudio (66,67)⁠. Specific libraries used for enrichment, clustering, and ontology analyses include STRINGdb (v2.6.0), ClusterProfiler (v4.2.0), EnrichGO (v4.2.0), EnrichPathway (v4.2.0), and Biomart (v2.50.1)⁠.
  
## Cheminformatics analysis

Python programming environment (v3.9.7) was used for chemical descriptor extraction, Morgan fingerprinting, Mol2vec fingerprinting, compound similarity assessment, substructure search, and image generation. Used packages and analytical frameworks include Rdkit (v2021.9.4),  numpy (v1.22.1), pandas (v1.3.5), seaborn (v0.11.2), matplotlib (v3.5.1), and chemexpy (v1.0.10). Custom algorithmic assessments, comparative analyses, and data mining were performed using Rdkit (v2021.9.4) as an analytical framework⁠.
  
## Machine and deep learning

Python programming environment (v3.9.7) was used for machine and deep learning. Scikit-learn was used for machine learning [LGBMClassifier](https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMClassifier.html) with default parameters, train-test split at 0.2, where features comprised vectorised and normalised Morgan fingerprints (radius=3, nBits=2048). Deep learning neural networks were built for [Mol2vec](https://pubs.acs.org/doi/10.1021/acs.jcim.7b00616) encoded chemical features using the following set-up: sequential addition of layers starting with a Dense layer (hidden units=200, activation=’relu’, and input shape=(300,)), followed by Dense layers with hidden units: 128, 100, 50 and a dropout of 0.25 after each. All layers except the last were activated with ‘relu’ function, the last dense layer had only 1 hidden unit and sigmoid activation. Binary cross-entropy with adam optimiser and metrics for accuracy were used for the model compilation. The analysis was run for 200 epochs using 256 units for batch size with 0.2 split of the original data for validation. Deep learning was performed using Python 3 Google Compute Engine backend (Tensor processing units, TPU), RAM 12 GB, and HDD 107 GB.
