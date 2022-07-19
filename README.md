<h1><b>Disease Prediction with GUI<b></h1>
    
A disease prediction model working on support vector machine (SVM). It takes the symptoms of the user as input along with its location and predicts the most probable disease which the user might be facing. The same data is being sent to cloud and being later analysed using analytical tool tableau.

For demonstration purpose, only the data of the diseases GERD and Hepatitis C is being sent to the cloud and analysed.

The data has been taken from https://www.kaggle.com/itachi9604/disease-symptom-description-dataset.

My Kaggle contribution for this dataset can be viewed at https://www.kaggle.com/code/kunal2350/disease-prediction-with-gui/notebook

<h2>Importing the libraries</h2>


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
import seaborn as sns
from tkinter import *
from tkinter import messagebox
import sys 
import urllib
import urllib.request
```

<h2>Importing the dataset</h2>


```python
df = pd.read_csv(r'C:\Users\kunal\Desktop\Github projects\DiseasePredictionwithGUI\dataset.csv')
print(df.head())
#df.describe()
df1 = pd.read_csv(r'C:\Users\kunal\Desktop\Github projects\DiseasePredictionwithGUI\Symptom-severity.csv')
print(df1.head())
```

                Disease   Symptom_1              Symptom_2              Symptom_3  \
    0  Fungal infection     itching              skin_rash   nodal_skin_eruptions   
    1  Fungal infection   skin_rash   nodal_skin_eruptions    dischromic _patches   
    2  Fungal infection     itching   nodal_skin_eruptions    dischromic _patches   
    3  Fungal infection     itching              skin_rash    dischromic _patches   
    4  Fungal infection     itching              skin_rash   nodal_skin_eruptions   
    
                  Symptom_4 Symptom_5 Symptom_6 Symptom_7 Symptom_8 Symptom_9  \
    0   dischromic _patches       NaN       NaN       NaN       NaN       NaN   
    1                   NaN       NaN       NaN       NaN       NaN       NaN   
    2                   NaN       NaN       NaN       NaN       NaN       NaN   
    3                   NaN       NaN       NaN       NaN       NaN       NaN   
    4                   NaN       NaN       NaN       NaN       NaN       NaN   
    
      Symptom_10 Symptom_11 Symptom_12 Symptom_13 Symptom_14 Symptom_15  \
    0        NaN        NaN        NaN        NaN        NaN        NaN   
    1        NaN        NaN        NaN        NaN        NaN        NaN   
    2        NaN        NaN        NaN        NaN        NaN        NaN   
    3        NaN        NaN        NaN        NaN        NaN        NaN   
    4        NaN        NaN        NaN        NaN        NaN        NaN   
    
      Symptom_16 Symptom_17  
    0        NaN        NaN  
    1        NaN        NaN  
    2        NaN        NaN  
    3        NaN        NaN  
    4        NaN        NaN  
                    Symptom  weight
    0               itching       1
    1             skin_rash       3
    2  nodal_skin_eruptions       4
    3   continuous_sneezing       4
    4             shivering       5
    

<h2>Cleaning of Data</h2>


```python
df.isna().sum()
df.isnull().sum()

cols = df.columns
data = df[cols].values.flatten()

s = pd.Series(data)
s = s.str.strip()
s = s.values.reshape(df.shape)

df = pd.DataFrame(s, columns=df.columns)

df = df.fillna(0)
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Disease</th>
      <th>Symptom_1</th>
      <th>Symptom_2</th>
      <th>Symptom_3</th>
      <th>Symptom_4</th>
      <th>Symptom_5</th>
      <th>Symptom_6</th>
      <th>Symptom_7</th>
      <th>Symptom_8</th>
      <th>Symptom_9</th>
      <th>Symptom_10</th>
      <th>Symptom_11</th>
      <th>Symptom_12</th>
      <th>Symptom_13</th>
      <th>Symptom_14</th>
      <th>Symptom_15</th>
      <th>Symptom_16</th>
      <th>Symptom_17</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Fungal infection</td>
      <td>itching</td>
      <td>skin_rash</td>
      <td>nodal_skin_eruptions</td>
      <td>dischromic _patches</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Fungal infection</td>
      <td>skin_rash</td>
      <td>nodal_skin_eruptions</td>
      <td>dischromic _patches</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Fungal infection</td>
      <td>itching</td>
      <td>nodal_skin_eruptions</td>
      <td>dischromic _patches</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Fungal infection</td>
      <td>itching</td>
      <td>skin_rash</td>
      <td>dischromic _patches</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Fungal infection</td>
      <td>itching</td>
      <td>skin_rash</td>
      <td>nodal_skin_eruptions</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



<h2>Encoding the the symptoms with their severity weight</h2>


```python
vals = df.values
symptoms = df1['Symptom'].unique()

for i in range(len(symptoms)):
    vals[vals == symptoms[i]] = df1[df1['Symptom'] == symptoms[i]]['weight'].values[0]
    
d = pd.DataFrame(vals, columns=cols)

d = d.replace('dischromic _patches', 0)
d = d.replace('spotting_ urination',0)
df = d.replace('foul_smell_of urine',0)
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Disease</th>
      <th>Symptom_1</th>
      <th>Symptom_2</th>
      <th>Symptom_3</th>
      <th>Symptom_4</th>
      <th>Symptom_5</th>
      <th>Symptom_6</th>
      <th>Symptom_7</th>
      <th>Symptom_8</th>
      <th>Symptom_9</th>
      <th>Symptom_10</th>
      <th>Symptom_11</th>
      <th>Symptom_12</th>
      <th>Symptom_13</th>
      <th>Symptom_14</th>
      <th>Symptom_15</th>
      <th>Symptom_16</th>
      <th>Symptom_17</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Fungal infection</td>
      <td>1</td>
      <td>3</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Fungal infection</td>
      <td>3</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Fungal infection</td>
      <td>1</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Fungal infection</td>
      <td>1</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Fungal infection</td>
      <td>1</td>
      <td>3</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



<h2> Storing the diseases and encoded symptoms in seperate dataframes</h2>


```python
(df[cols] == 0).all()

df['Disease'].value_counts()

df['Disease'].unique()

data = df.iloc[:,1:].values
labels = df['Disease'].values
```

<h2>Splitting the data and training the model</h2>


```python
x_train, x_test, y_train, y_test = train_test_split(data, labels, shuffle=True, train_size = 0.85)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

model = SVC()
model.fit(x_train, y_train)

preds = model.predict(x_test)
print(preds)
```

    (4182, 17) (738, 17) (4182,) (738,)
    ['Allergy' 'Typhoid' 'Psoriasis' 'Urinary tract infection' 'Tuberculosis'
     'Pneumonia' 'Hepatitis B' 'Typhoid' 'Osteoarthristis' 'Hypothyroidism'
     'Urinary tract infection' 'Arthritis' 'Tuberculosis' 'Gastroenteritis'
     'Diabetes' 'Migraine' 'GERD' '(vertigo) Paroymsal  Positional Vertigo'
     'hepatitis A' 'Hepatitis B' 'Malaria' 'Dimorphic hemmorhoids(piles)'
     'Psoriasis' 'Fungal infection' 'Hepatitis D' 'Dengue' 'Chicken pox'
     'Hepatitis B' '(vertigo) Paroymsal  Positional Vertigo' 'Chicken pox'
     'Hepatitis B' 'Dengue' 'Hepatitis C' 'Typhoid' 'Chronic cholestasis'
     'AIDS' 'Hepatitis E' 'Chronic cholestasis' 'Paralysis (brain hemorrhage)'
     '(vertigo) Paroymsal  Positional Vertigo' 'Dimorphic hemmorhoids(piles)'
     'Chronic cholestasis' '(vertigo) Paroymsal  Positional Vertigo'
     'Hepatitis D' 'Peptic ulcer diseae' 'Hepatitis B' 'Heart attack'
     'Typhoid' 'Alcoholic hepatitis' 'GERD' 'AIDS' 'Dengue' 'Typhoid'
     'Hypertension' 'Acne' 'Migraine' 'Peptic ulcer diseae' 'Arthritis'
     'Fungal infection' 'Paralysis (brain hemorrhage)' 'Hypoglycemia'
     'Paralysis (brain hemorrhage)' 'Chicken pox' 'hepatitis A' 'Dengue'
     'Hyperthyroidism' 'Allergy' 'Impetigo' 'Gastroenteritis' 'Hypertension'
     'Varicose veins' 'Diabetes' 'Tuberculosis' 'Hypothyroidism' 'Allergy'
     'Arthritis' 'Gastroenteritis' 'Hepatitis B' 'Hypothyroidism'
     'Hepatitis B' 'Peptic ulcer diseae'
     '(vertigo) Paroymsal  Positional Vertigo'
     '(vertigo) Paroymsal  Positional Vertigo' 'Tuberculosis' 'Impetigo'
     'Hyperthyroidism' 'Varicose veins'
     '(vertigo) Paroymsal  Positional Vertigo' 'Impetigo'
     'Alcoholic hepatitis' 'hepatitis A' 'Hypoglycemia' 'Hepatitis C'
     'Typhoid' 'Hypoglycemia' 'Dimorphic hemmorhoids(piles)' 'Psoriasis'
     'Osteoarthristis' 'Chicken pox' 'Hypothyroidism' 'Alcoholic hepatitis'
     'Hepatitis D' 'Hepatitis B' 'Fungal infection' 'Hepatitis C'
     'Alcoholic hepatitis' 'Psoriasis' 'Chronic cholestasis' 'Hypothyroidism'
     'Hepatitis B' 'Hypertension' 'Psoriasis' 'Migraine' 'Hepatitis E'
     'Psoriasis' 'Gastroenteritis' 'GERD' 'Hypothyroidism'
     'Paralysis (brain hemorrhage)' 'Psoriasis' 'Cervical spondylosis'
     'Hepatitis B' 'Paralysis (brain hemorrhage)' 'Alcoholic hepatitis'
     'Pneumonia' 'Fungal infection' 'Hypothyroidism' 'Drug Reaction'
     'Tuberculosis' 'Hyperthyroidism' 'hepatitis A' 'Hypoglycemia' 'Pneumonia'
     'Hepatitis C' 'Gastroenteritis' 'Acne' 'Hepatitis B' 'Bronchial Asthma'
     'Hypoglycemia' 'Tuberculosis' 'Migraine' 'Arthritis' 'Hypothyroidism'
     'Tuberculosis' 'Drug Reaction' 'Allergy'
     '(vertigo) Paroymsal  Positional Vertigo' 'Common Cold' 'Malaria'
     'Allergy' 'Drug Reaction' 'Chicken pox' 'Urinary tract infection'
     'Common Cold' 'Diabetes' 'Hepatitis D' 'Chronic cholestasis'
     'Heart attack' 'Drug Reaction' 'Paralysis (brain hemorrhage)'
     'Peptic ulcer diseae' 'Heart attack' 'Fungal infection' 'AIDS'
     'Hepatitis E' 'Cervical spondylosis' 'Hypothyroidism'
     '(vertigo) Paroymsal  Positional Vertigo'
     '(vertigo) Paroymsal  Positional Vertigo' 'Peptic ulcer diseae'
     'Alcoholic hepatitis' '(vertigo) Paroymsal  Positional Vertigo'
     'Fungal infection' 'Heart attack' 'Bronchial Asthma' 'Dengue'
     'Hepatitis D' 'Hyperthyroidism' 'Dengue' 'Bronchial Asthma'
     'Dimorphic hemmorhoids(piles)' 'Cervical spondylosis' 'Gastroenteritis'
     'Chicken pox' 'Diabetes' 'Peptic ulcer diseae' 'Chronic cholestasis'
     'Gastroenteritis' 'Chicken pox' 'GERD' 'Hypothyroidism'
     'Chronic cholestasis' 'Urinary tract infection' 'Arthritis' 'Arthritis'
     '(vertigo) Paroymsal  Positional Vertigo' 'Acne' 'Common Cold'
     'Hypertension' 'Osteoarthristis' 'Hyperthyroidism' 'Malaria' 'Acne'
     'Dengue' 'Hepatitis E' 'Hepatitis B' 'Hypoglycemia' 'Diabetes'
     'Peptic ulcer diseae' 'Hypoglycemia' 'Common Cold' 'Typhoid'
     'Fungal infection' 'Hypothyroidism' 'Varicose veins' 'AIDS' 'AIDS'
     'Hepatitis E' 'Urinary tract infection' 'Tuberculosis' 'Malaria'
     'Fungal infection' 'Impetigo' 'Peptic ulcer diseae' 'Allergy' 'GERD'
     'Gastroenteritis' '(vertigo) Paroymsal  Positional Vertigo' 'Migraine'
     'Hypoglycemia' 'Fungal infection' 'Cervical spondylosis'
     'Chronic cholestasis' 'Hepatitis E' 'Migraine' 'Psoriasis'
     'Chronic cholestasis' 'Cervical spondylosis' 'Impetigo' 'Psoriasis'
     'Dimorphic hemmorhoids(piles)' 'Migraine' 'Cervical spondylosis'
     'Migraine' 'Alcoholic hepatitis' 'Psoriasis' 'Chronic cholestasis'
     'Hypothyroidism' 'AIDS' 'GERD' 'Typhoid'
     '(vertigo) Paroymsal  Positional Vertigo' 'Hypothyroidism' 'Tuberculosis'
     'Hyperthyroidism' 'Heart attack' 'GERD' 'Acne'
     'Paralysis (brain hemorrhage)' 'Hepatitis C' 'Typhoid' 'Common Cold'
     'Gastroenteritis' 'Dengue' 'Urinary tract infection' 'Hyperthyroidism'
     'Arthritis' 'Common Cold' 'Psoriasis' 'Hypertension' 'Hepatitis C'
     'Hepatitis D' 'hepatitis A' 'Acne' 'Hepatitis E' 'Hepatitis C'
     'Arthritis' 'Peptic ulcer diseae' 'Alcoholic hepatitis' 'Heart attack'
     'Malaria' 'Alcoholic hepatitis' 'Heart attack' 'Gastroenteritis'
     'Peptic ulcer diseae' 'Typhoid' 'Tuberculosis' 'Typhoid' 'Hypertension'
     'Heart attack' 'Acne' 'Dimorphic hemmorhoids(piles)'
     'Cervical spondylosis' 'Pneumonia' 'Bronchial Asthma' 'Drug Reaction'
     'Psoriasis' 'GERD' 'Impetigo' 'Hypertension' 'Peptic ulcer diseae'
     'Pneumonia' 'Heart attack' 'Dimorphic hemmorhoids(piles)'
     'Urinary tract infection' 'Hyperthyroidism'
     'Dimorphic hemmorhoids(piles)' 'Hypertension' 'GERD' 'Drug Reaction'
     '(vertigo) Paroymsal  Positional Vertigo' 'Cervical spondylosis' 'Acne'
     'Tuberculosis' 'Osteoarthristis' 'Hepatitis E' 'Tuberculosis'
     'Paralysis (brain hemorrhage)' 'Hepatitis C' 'Osteoarthristis'
     'Arthritis' 'Hepatitis E' 'Diabetes' 'Hepatitis E'
     '(vertigo) Paroymsal  Positional Vertigo' 'Acne' 'Arthritis'
     'Gastroenteritis' 'Jaundice' '(vertigo) Paroymsal  Positional Vertigo'
     'Arthritis' 'Drug Reaction' 'Varicose veins' 'Hepatitis D' 'Common Cold'
     '(vertigo) Paroymsal  Positional Vertigo' 'Bronchial Asthma'
     'Hypoglycemia' 'Allergy' 'Fungal infection' 'Alcoholic hepatitis'
     'Hypoglycemia' 'Alcoholic hepatitis' 'Psoriasis' 'Tuberculosis' 'Dengue'
     'Diabetes' 'Malaria' 'Hepatitis C' 'Arthritis' 'Malaria'
     'Alcoholic hepatitis' 'Hyperthyroidism'
     '(vertigo) Paroymsal  Positional Vertigo' 'Malaria' 'Migraine'
     'Psoriasis' 'Drug Reaction' 'Fungal infection' 'Common Cold'
     'Gastroenteritis' 'Dengue' 'Bronchial Asthma' 'Fungal infection'
     '(vertigo) Paroymsal  Positional Vertigo' 'Gastroenteritis'
     'Cervical spondylosis' 'Malaria' 'Peptic ulcer diseae' 'Gastroenteritis'
     'Common Cold' 'Typhoid' 'Bronchial Asthma' 'Heart attack' 'Hepatitis B'
     'Allergy' 'Allergy' 'Dimorphic hemmorhoids(piles)' 'Typhoid' 'AIDS'
     'Hepatitis E' 'Hypothyroidism' 'Alcoholic hepatitis' 'Gastroenteritis'
     'Osteoarthristis' 'Urinary tract infection' 'Gastroenteritis' 'Allergy'
     'Hypothyroidism' 'Paralysis (brain hemorrhage)'
     'Dimorphic hemmorhoids(piles)' 'Malaria' 'Varicose veins'
     'Paralysis (brain hemorrhage)' 'hepatitis A' 'Dengue' 'Common Cold'
     'AIDS' 'Hepatitis B' 'Diabetes' 'Bronchial Asthma' 'Psoriasis'
     'Drug Reaction' 'Hepatitis B' 'Hepatitis C' 'Hepatitis C'
     'Osteoarthristis' 'Chicken pox' 'Hepatitis E' 'Hepatitis B' 'Migraine'
     'Cervical spondylosis' 'Hyperthyroidism' 'GERD' 'Varicose veins'
     'Paralysis (brain hemorrhage)' 'Peptic ulcer diseae' 'Migraine'
     'Osteoarthristis' 'Impetigo' 'Arthritis' 'Gastroenteritis'
     'Drug Reaction' 'Hepatitis E' 'Allergy' 'hepatitis A'
     'Dimorphic hemmorhoids(piles)' 'Acne' 'Hyperthyroidism' 'Hypoglycemia'
     'Paralysis (brain hemorrhage)' 'Common Cold' 'Malaria' 'Impetigo'
     'Heart attack' 'Urinary tract infection' 'Hypothyroidism' 'AIDS'
     'Fungal infection' 'Chronic cholestasis' 'Common Cold'
     'Dimorphic hemmorhoids(piles)' 'hepatitis A' 'Diabetes' 'Varicose veins'
     'Drug Reaction' 'Acne' 'Tuberculosis' 'GERD' 'Hyperthyroidism'
     'Urinary tract infection' 'Tuberculosis' 'Dimorphic hemmorhoids(piles)'
     'Hypothyroidism' 'Impetigo' 'AIDS' 'GERD' 'Hepatitis C' 'Migraine' 'GERD'
     'Malaria' 'Varicose veins' 'GERD' 'Chronic cholestasis' 'Tuberculosis'
     'Chronic cholestasis' 'Dengue' 'Arthritis' 'Hypertension' 'Impetigo'
     'Fungal infection' 'Hepatitis B' 'Jaundice' 'Chicken pox' 'Malaria'
     'Acne' 'Impetigo' 'hepatitis A' 'Cervical spondylosis' 'Psoriasis'
     'Fungal infection' 'Drug Reaction' 'Urinary tract infection' 'Arthritis'
     'Pneumonia' 'Paralysis (brain hemorrhage)' 'Malaria' 'Drug Reaction'
     'hepatitis A' 'Hypothyroidism' 'Jaundice' 'AIDS' 'Dengue' 'Pneumonia'
     'Osteoarthristis' 'AIDS' 'Hepatitis D' 'Impetigo' 'Hepatitis E'
     'Hyperthyroidism' 'Hypoglycemia' 'Allergy' 'Alcoholic hepatitis'
     'Psoriasis' 'Hypoglycemia' 'Chicken pox' 'Paralysis (brain hemorrhage)'
     'Chronic cholestasis' 'Hypertension' 'AIDS' 'Jaundice' 'Varicose veins'
     'Urinary tract infection' 'Cervical spondylosis' 'Common Cold' 'GERD'
     'Hepatitis E' 'Bronchial Asthma' 'Chronic cholestasis' 'Acne'
     'Gastroenteritis' 'Migraine' 'Dengue' 'Arthritis' 'Psoriasis'
     'Hyperthyroidism' 'Hypothyroidism' 'Hypertension' 'AIDS' 'hepatitis A'
     'Migraine' 'Cervical spondylosis' 'Osteoarthristis' 'Acne'
     'Paralysis (brain hemorrhage)' 'Arthritis' 'Dimorphic hemmorhoids(piles)'
     'Pneumonia' 'Heart attack' 'Fungal infection' 'Chicken pox' 'hepatitis A'
     'Hepatitis D' 'hepatitis A' 'Common Cold' 'Hepatitis E' 'Gastroenteritis'
     'Gastroenteritis' 'Diabetes' 'Peptic ulcer diseae' 'Hepatitis C'
     'Tuberculosis' 'Psoriasis' 'Chronic cholestasis' 'Osteoarthristis' 'GERD'
     'Alcoholic hepatitis' 'Peptic ulcer diseae' 'Peptic ulcer diseae'
     'Gastroenteritis' 'Hepatitis E' 'Hyperthyroidism' 'Alcoholic hepatitis'
     'Gastroenteritis' 'Heart attack' 'Acne' 'Gastroenteritis'
     'Chronic cholestasis' 'Heart attack' 'Psoriasis' 'Alcoholic hepatitis'
     '(vertigo) Paroymsal  Positional Vertigo' 'Osteoarthristis'
     'Chronic cholestasis' 'Impetigo' 'Psoriasis' 'Diabetes'
     'Peptic ulcer diseae' 'Hypothyroidism' 'Paralysis (brain hemorrhage)'
     'Psoriasis' 'Acne' 'Paralysis (brain hemorrhage)' 'Diabetes'
     'Hypoglycemia' 'hepatitis A' 'AIDS'
     '(vertigo) Paroymsal  Positional Vertigo' 'Drug Reaction' 'Psoriasis'
     'hepatitis A' 'Hepatitis B' 'Hepatitis E' 'Hepatitis E' 'Hypertension'
     'Allergy' 'Hepatitis C' 'Peptic ulcer diseae' 'Alcoholic hepatitis'
     'GERD' 'Hypothyroidism' 'Cervical spondylosis'
     'Dimorphic hemmorhoids(piles)' 'Pneumonia' 'Hepatitis D'
     'Bronchial Asthma' 'Varicose veins' 'Migraine' 'Hypoglycemia'
     'Fungal infection' 'Hepatitis E' 'Dengue' 'Hepatitis D' 'Gastroenteritis'
     'AIDS' 'Osteoarthristis' 'Drug Reaction' 'Peptic ulcer diseae'
     'Alcoholic hepatitis' 'Arthritis'
     '(vertigo) Paroymsal  Positional Vertigo' 'Migraine'
     'Urinary tract infection' 'Chicken pox' 'Dengue' 'Typhoid' 'Malaria'
     'Jaundice' 'Hepatitis E' 'Chronic cholestasis' 'Hepatitis B' 'Jaundice'
     'Common Cold' 'Diabetes' 'Fungal infection' 'Dengue' 'Hypoglycemia'
     'Dengue' 'Arthritis' 'Allergy' 'Fungal infection' 'Impetigo'
     'Fungal infection' 'Dengue' 'Diabetes' 'Migraine' 'Bronchial Asthma'
     'Acne' 'Gastroenteritis' 'Typhoid' 'Dimorphic hemmorhoids(piles)'
     'Varicose veins' 'Dimorphic hemmorhoids(piles)' 'Diabetes' 'Hepatitis B'
     'Hyperthyroidism' 'Paralysis (brain hemorrhage)' 'Typhoid'
     '(vertigo) Paroymsal  Positional Vertigo' 'Osteoarthristis' 'hepatitis A'
     'Hepatitis C' 'Common Cold' 'Peptic ulcer diseae' 'Hepatitis C'
     'Arthritis' 'Migraine' 'Paralysis (brain hemorrhage)' 'Jaundice' 'Dengue'
     'Chronic cholestasis' 'Hypertension' 'Dengue' 'Chicken pox' 'Hepatitis B'
     '(vertigo) Paroymsal  Positional Vertigo' 'Hepatitis D' 'Hepatitis E'
     'Hepatitis D' 'Gastroenteritis' 'Chicken pox' 'Diabetes' 'Malaria'
     'Migraine' '(vertigo) Paroymsal  Positional Vertigo'
     'Paralysis (brain hemorrhage)' 'Chronic cholestasis' 'Psoriasis'
     '(vertigo) Paroymsal  Positional Vertigo' 'Malaria' 'Jaundice' 'AIDS'
     'GERD' 'Hepatitis E' 'Typhoid' 'Hyperthyroidism' 'Pneumonia'
     'Cervical spondylosis' 'Psoriasis' 'Common Cold' 'Heart attack'
     'Heart attack' 'Diabetes' 'Hypoglycemia' 'hepatitis A' 'AIDS'
     'Tuberculosis' 'Gastroenteritis' 'Peptic ulcer diseae' 'Migraine'
     'Hepatitis D' 'Drug Reaction' 'Chronic cholestasis' 'AIDS' 'Hypoglycemia'
     'Allergy' 'Allergy' 'Urinary tract infection' 'Hyperthyroidism'
     'Arthritis' 'Dimorphic hemmorhoids(piles)' 'Hepatitis B' 'Hepatitis C'
     'Chicken pox' 'Arthritis' 'Hypertension' 'Hepatitis B'
     'Peptic ulcer diseae' 'Impetigo' 'Urinary tract infection'
     '(vertigo) Paroymsal  Positional Vertigo' 'Hyperthyroidism'
     'Fungal infection' 'Diabetes' '(vertigo) Paroymsal  Positional Vertigo'
     'Dimorphic hemmorhoids(piles)' 'Hepatitis C' 'Hepatitis C']
    

<h2>Checking accuracy of the model</h2>


```python
conf_mat = confusion_matrix(y_test, preds)
df_cm = pd.DataFrame(conf_mat, index=df['Disease'].unique(), columns=df['Disease'].unique())
print('F1-score% =', f1_score(y_test, preds, average='macro')*100, '|', 'Accuracy% =', accuracy_score(y_test, preds)*100)
sns.heatmap(df_cm)
```

    F1-score% = 93.93014228175568 | Accuracy% = 94.3089430894309
    




    <matplotlib.axes._subplots.AxesSubplot at 0x220dab8b6d8>




![png](output_16_2.png)


<h2>Functions used for prediction of user inputs and sending data to cloud</h2>


```python
def message():
    if (Symptom1.get() == "None" and  Symptom2.get() == "None" and Symptom3.get() == "None" and Symptom4.get() == "None" and Symptom5.get() == "None"):
        messagebox.showinfo("OPPS!!", "ENTER  SYMPTOMS PLEASE")
    else :
        SVM()

def SVM():
    psymptoms = [Symptom1.get(),Symptom2.get(),Symptom3.get(),Symptom4.get(),Symptom5.get()]
    loc = location.get()
    a = np.array(df1["Symptom"])
    b = np.array(df1["weight"])
    for j in range(len(psymptoms)):
        for k in range(len(a)):
            if psymptoms[j]==a[k]:
                psymptoms[j]=b[k]

    nulls = [0,0,0,0,0,0,0,0,0,0,0,0]
    psy = [psymptoms + nulls]

    pred2 = model.predict(psy)
    t3.delete("1.0", END)
    t3.insert(END, pred2[0])

    if(pred2[0]=="GERD"):
        z=urllib.request.urlopen('https://api.thingspeak.com/update?api_key=MP77HD9B13Z7N6BO&field1=1&field2=0&field3='+str(loc))
        z.read()
    if(pred2[0]=="Hepatitis C"):
        r=urllib.request.urlopen('https://api.thingspeak.com/update?api_key=MP77HD9B13Z7N6BO&field1=0&field2=1&field3='+str(loc))
        r.read()
```

<p>Thingspeak is being used as the cloud. For simplicity, only when the diseases GERD or Hepatitis C are detected, the data is sent to cloud. The datas sent to cloud are the predicted disease and the location of the user.</p>
<p>Every time GERD is predicted, 1 is sent to GERD field and 0 to Hepatitis field and vice versa if Hepatitis is detected.
1 and 0 has been choosen for ease in aggrevation while analytics.<p>

<h2>Designing of GUI</h2>


```python
root = Tk()
root.title(" Disease Prediction From Symptoms")
root.configure()

Symptom1 = StringVar()
Symptom1.set(None)
Symptom2 = StringVar()
Symptom2.set(None)
Symptom3 = StringVar()
Symptom3.set(None)
Symptom4 = StringVar()
Symptom4.set(None)
Symptom5 = StringVar()
Symptom5.set(None)
location = StringVar()
location.set(None)

w2 = Label(root, justify=CENTER, text=" Disease Prediction From Symptoms ")
w2.config(font=("Helvetica", 30))
w2.grid(row=1, column=0, columnspan=2, padx=100)

NameLb1 = Label(root, text="")
NameLb1.config(font=("Helvetica", 20))
NameLb1.grid(row=5, column=1, pady=10,  sticky=W)

S1Lb = Label(root,  text="Symptom 1")
S1Lb.config(font=("Helvetica", 15))
S1Lb.grid(row=7, column=1, pady=10 , sticky=W)

S2Lb = Label(root,  text="Symptom 2")
S2Lb.config(font=("Helvetica", 15))
S2Lb.grid(row=8, column=1, pady=10, sticky=W)

S3Lb = Label(root,  text="Symptom 3")
S3Lb.config(font=("Helvetica", 15))
S3Lb.grid(row=9, column=1, pady=10, sticky=W)

S4Lb = Label(root,  text="Symptom 4")
S4Lb.config(font=("Helvetica", 15))
S4Lb.grid(row=10, column=1, pady=10, sticky=W)

S5Lb = Label(root,  text="Symptom 5")
S5Lb.config(font=("Helvetica", 15))
S5Lb.grid(row=11, column=1, pady=10, sticky=W)

locLb = Label(root,  text="Location")
locLb.config(font=("Helvetica", 15))
locLb.grid(row=12, column=1, pady=10, sticky=W)

lr = Button(root, text="Predict",height=2, width=20, command=message)
lr.config(font=("Helvetica", 15))
lr.grid(row=15, column=1,pady=10)

#OPTIONS = sorted(symptoms)
OPTIONS = ["fatigue", "yellowish_skin", "loss_of_appetite", "yellowing_of_eyes", 'family_history',"stomach_pain", "ulcers_on_tongue", "vomiting", "cough", "chest_pain"]
LOCATIONS = ["New Delhi", "Mumbai", "Chennai", "Kolkata", "Bengaluru"]

S1En = OptionMenu(root, Symptom1,*OPTIONS)
S1En.grid(row=7, column=1)

S2En = OptionMenu(root, Symptom2,*OPTIONS)
S2En.grid(row=8, column=1)

S3En = OptionMenu(root, Symptom3,*OPTIONS)
S3En.grid(row=9, column=1)

S4En = OptionMenu(root, Symptom4,*OPTIONS)
S4En.grid(row=10, column=1)

S5En = OptionMenu(root, Symptom5,*OPTIONS)
S5En.grid(row=11, column=1)

LocEn = OptionMenu(root, location,*LOCATIONS)
LocEn.grid(row=12, column=1)

NameLb = Label(root, text="")
NameLb.config(font=("Helvetica", 20))
NameLb.grid(row=13, column=1, pady=10,  sticky=W)

NameLb = Label(root, text="")
NameLb.config(font=("Helvetica", 15))
NameLb.grid(row=18, column=1, pady=10,  sticky=W)

t3 = Text(root, height=2, width=20)
t3.config(font=("Helvetica", 20))
t3.grid(row=19, column=1 , padx=10)

root.mainloop()
```

    Exception in Tkinter callback
    Traceback (most recent call last):
      File "D:\Anaconda3\envs\tf\lib\tkinter\__init__.py", line 1705, in __call__
        return self.func(*args)
      File "<ipython-input-17-707dc1adc7eb>", line 5, in message
        SVM()
      File "<ipython-input-17-707dc1adc7eb>", line 28, in SVM
        r=urllib.request.urlopen('https://api.thingspeak.com/update?api_key=MP77HD9B13Z7N6BO&field1=0&field2=1&field3='+str(loc))
      File "D:\Anaconda3\envs\tf\lib\urllib\request.py", line 223, in urlopen
        return opener.open(url, data, timeout)
      File "D:\Anaconda3\envs\tf\lib\urllib\request.py", line 526, in open
        response = self._open(req, data)
      File "D:\Anaconda3\envs\tf\lib\urllib\request.py", line 544, in _open
        '_open', req)
      File "D:\Anaconda3\envs\tf\lib\urllib\request.py", line 504, in _call_chain
        result = func(*args)
      File "D:\Anaconda3\envs\tf\lib\urllib\request.py", line 1361, in https_open
        context=self._context, check_hostname=self._check_hostname)
      File "D:\Anaconda3\envs\tf\lib\urllib\request.py", line 1318, in do_open
        encode_chunked=req.has_header('Transfer-encoding'))
      File "D:\Anaconda3\envs\tf\lib\http\client.py", line 1262, in request
        self._send_request(method, url, body, headers, encode_chunked)
      File "D:\Anaconda3\envs\tf\lib\http\client.py", line 1273, in _send_request
        self.putrequest(method, url, **skips)
      File "D:\Anaconda3\envs\tf\lib\http\client.py", line 1124, in putrequest
        self._validate_path(url)
      File "D:\Anaconda3\envs\tf\lib\http\client.py", line 1215, in _validate_path
        raise InvalidURL(f"URL can't contain control characters. {url!r} "
    http.client.InvalidURL: URL can't contain control characters. '/update?api_key=MP77HD9B13Z7N6BO&field1=0&field2=1&field3=New Delhi' (found at least ' ')
    



The second list named "OPTIONS" which has been commented are the symptoms of GERD and Hepatitis C. They can be directly used if needed.

![image.png](attachment:image.png)

GUI used for taking user input. The drop downs show the list of symptoms from which the user can choose.

<h2>Output Predicted and sent to Thingspeak</h2>

![image.png](attachment:image.png)

![image.png](attachment:image.png)

<h2>Analytics Dashboard using Tableau software</h2>

![image.png](attachment:image.png)

This dashboard and indivisual sheets can be viewed at https://public.tableau.com/views/DiseaseDashboard_16216759756860/Dashboard1?:language=en&:display_count=y&:origin=viz_share_link


```python

```
