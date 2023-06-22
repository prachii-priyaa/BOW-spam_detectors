import sys
import numpy as np
import pandas as pd
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB

# Function to train the model
def train_model(train_file, model_file):
    # Load training data from CSV file
    train_data = pd.read_csv(train_file)

    # Preprocessing
    train_data['Category'] = train_data['Category'].str.lower()
    train_data['Category'] = train_data['Category'].str.replace('[^\w\s]', '', regex=True)

    # Create bag-of-words representation of the text data
    vectorizer = CountVectorizer()
    X_train = vectorizer.fit_transform(train_data['Category'])
    y_train = np.where(train_data['Message'] == 'spam', 1, 0)

    # Train a Naive Bayes classifier
    clf = MultinomialNB()
    clf.fit(X_train, y_train)

    # Save the trained model and vectorizer using pickle
    with open(model_file, 'wb') as f:
        pickle.dump(clf, f)
        pickle.dump(vectorizer, f)

    print('Model trained successfully and saved to', model_file)


# Function to test the dataset against an existing model
def test_model(test_file, model_file):
    # Load test data from CSV file
    test_data = pd.read_csv(test_file)

    # Preprocessing
    test_data['Category'] = test_data['Category'].str.lower()
    test_data['Category'] = test_data['Category'].str.replace('[^\w\s]', '', regex=True)

    # Load the trained model and vectorizer
    with open(model_file, 'rb') as f:
        clf = pickle.load(f)
        vectorizer = pickle.load(f)

    # Create bag-of-words representation of the text data
    X_test = vectorizer.transform(test_data['Category'])
    y_test = np.where(test_data['Message'] == 'spam', 1, 0)

    # Make predictions using the model
    y_pred = clf.predict(X_test)

    # Evaluate the model on the test set
    accuracy = accuracy_score(y_test, y_pred)
    print('Accuracy:', accuracy)


# Check command line arguments
if len(sys.argv) < 4:
    print('Insufficient arguments!')
    print('Usage: python spam_detection.py train <train_file.csv> <model_file.pkl>')
    print('       python spam_detection.py test <test_file.csv> <model_file.pkl>')
    sys.exit(1)

# Read command line arguments
mode = sys.argv[1]
input_file = sys.argv[2]
model_file = sys.argv[3]

if mode == 'train':
    # Train a new model
    print('Training the model...')
    train_model(input_file, model_file)
elif mode == 'test':
    # Test dataset against an existing model
    print('Testing the model...')
    test_model(input_file, model_file)
elif mode == 'load':
    # Load a pre-trained model
    print('Loading the model...')
    test_model(input_file, model_file)
else:
    print('Invalid mode! Mode must be either "train", "test", or "load".')
    sys.exit(1)
