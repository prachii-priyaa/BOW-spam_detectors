#!/usr/bin/env python
# coding: utf-8



import sys
import numpy as np
import pandas as pd
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB




# Load data from CSV file
data = pd.read_csv(sys.argv[1])

# Preprocessing
data['Category'] = data['Category'].str.lower()
data['Category'] = data['Category'].str.replace('[^\w\s]', '')

# Split data into training and testing sets
train_data = data.sample(frac=0.8, random_state=1)
test_data = data.drop(train_data.index)

# Create bag-of-words representation of the text data
vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(train_data['Category'])
X_test = vectorizer.transform(test_data['Category'])

# Create labels
y_train = np.where(train_data['Message'] == 'spam', 1, 0)
y_test = np.where(test_data['Message'] == 'spam', 1, 0)

# Train a Naive Bayes classifier
clf = MultinomialNB()
clf.fit(X_train, y_train)

# Evaluate the model on the test set
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)

# Save the trained model using pickle
with open('spam_detection_model.pkl', 'wb') as f:
    pickle.dump(clf, f)
    pickle.dump(vectorizer, f)






