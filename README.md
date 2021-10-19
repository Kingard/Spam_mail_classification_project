
# Spam email classification using MLP, RandomForest and SVM

A spam email classification project implemented with a deep 
learning model (MLP) along side two other machine learning models
(RandomForest and SVM). It identifies potential spam messages and
can be integrated as a spam filter.


## Badges
![Jupyter](https://img.shields.io/badge/Python-Jupyter-orange)
![SVM](https://img.shields.io/badge/ML-SVM-lightgrey)
![Random Forest](https://img.shields.io/badge/ML-RandomForest%20-blue)
![Deep learning](https://img.shields.io/badge/Deep%20Learning-MLP-red)
  
 ###### badges made from: [shields.io](https://shields.io/)
 
## Dependency list

* Numpy
* Pandas
* Scikit learn
* Matplotlib
* Requests
* Pipeline
* TF-IDF


  
## Work flow

- Obtaining the data from: [spam data](https://raw.githubusercontent.com/laxmimerit/All-CSV-ML-Data-Files-Download/master/spam.tsv)
The data was obtained using the `requests` library and then passed
into a `pandas` dataframe.

- Data preparation
The data was carefully inspected for null values and imbalance in 
data categories. By convention unbalanced datasets don't perform 
well.


- Exploratory Data Analysis
This was done to observe how variables llike length of text and 
punction marks are distributed in spam (unwanted mails) and ham 
(desired mails).


- Word embedding and modeling
The text was vectorized using `TF-IDF` and the `Pipeline` library
was used to schedule the embedding and modeling.

- Model evaluation
The models were evaluated for performance and also tested with
random words



  

  