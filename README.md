

Libs that must be downloaded

import nltk
import numpy
import pandas
import regex

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier


#----->>>>>>>>> pip install nltk numpy pandas regex scikit-learn




| Library        | Used for                         |
| -------------- | -------------------------------- |
| `nltk`         | Text tokenization + stopwords    |
| `numpy`        | Array saving (e.g. predictions)  |
| `pandas`       | Reading and processing CSV files |
| `regex`        | URL and text cleaning            |
| `scikit-learn` | ML models + pipeline + TF-IDF    |



###############################################################################################################


main.py main file


###############################################################################################################


test.csv-> add text that you want to be predicted

prediction_<Model_Name> basically you dont need need, just predictions based an 30% of train_emotion

preprocessed_data - data that algorithm see after preprocessing

predicted_emotions file with predictions of test.csv

##############################################################################################################

By CoffiZ
