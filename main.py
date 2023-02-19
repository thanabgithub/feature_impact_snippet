import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# load the dataset
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data'
names = ['class', 'alcohol', 'malic_acid', 'ash', 'alcalinity_of_ash',
         'magnesium', 'total_phenols', 'flavanoids', 'nonflavanoid_phenols',
         'proanthocyanins', 'color_intensity', 'hue', 'od280_od315_of_diluted_wines', 'proline']
df = pd.read_csv(url, names=names)

# set the target variable and the features
y = df['class']
X = df.drop('class', axis=1)

# split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# create the model and fit to the training data
model = LogisticRegression()
model.fit(X_train, y_train)

# compute the performance on the test set
score = model.score(X_test, y_test)
print("Full model accuracy:", score)

# loop through each feature and remove it from the dataset
for feature in X_train.columns:
    X_reduced = X_train.drop(feature, axis=1)
    X_test_reduced = X_test.drop(feature, axis=1)

    # train a new model on the reduced dataset
    model_reduced = LogisticRegression()
    model_reduced.fit(X_reduced, y_train)

    # compute the performance of the reduced model on the test set
    score_reduced = model_reduced.score(X_test_reduced, y_test)

    # compute the feature impact as the difference between the full model accuracy and the reduced model accuracy
    impact = score - score_reduced
    print("Feature:", feature, "Impact:", impact)
