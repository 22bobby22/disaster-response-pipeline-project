import sys
import pandas as pd
import numpy as np
import re

import sqlalchemy
from sqlalchemy import create_engine

import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer

import sklearn
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report

import pickle

def load_data(database_filepath):
    """
    Input:
        database_filepath: path of the database from which to extract the data
    
    Extracts data from the database and selects the necessary data to create the model.
    
    Output:
        X: contains the column with the text of the messages which will be used to train and analyze
        Y: contains the columns with of the 36 categories which will be used to train and predict
        category_names: list with the names of the 36 categories
    """
    # create database connection
    engine = create_engine('sqlite:///' + database_filepath)
    
    # extract data and load to dataframe
    df = pd.read_sql_table('Messages', engine)
    
    # select the data needed to create the model
    X = df['message']
    Y = df[df.columns[4:]]
    category_names = df.columns[4:]
    return X, Y, category_names


def tokenize(text):
    """
    Input:
        text: message from the database
    
    Processes the text provided and tokenizes it to reduce complexity.
    
    Output:
        tokens: list of clean tokens
    """
    # transform text to lowercase
    text = text.lower()
    
    # remove punctuation marks
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)
    
    # create list of tokens
    tokens = word_tokenize(text)
    
    # remove stopwords
    tokens = [t for t in tokens if t not in stopwords.words("english")]
    
    # get the root form of the words
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return tokens


def build_model():
    """
    Creates the model which is a GridSearchCV object.
    
    Ouput:
        model: model that will be trained and used to predict the categories of the messages
    """
    # create pipeline
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    # select parameters for the grid search
    parameters = {
        'clf__estimator__n_estimators': [10, 20],
        'clf__estimator__max_depth': [2, None],
        'clf__estimator__min_samples_leaf': [1, 5, 10]
    }
    
    # create GridSearchCV object
    cv = GridSearchCV(pipeline, param_grid=parameters, n_jobs=4, verbose=2)
    
    return cv


def evaluate_model(model, X_test, Y_test):
    """
    Input:
        model: model used to predict
        X_test: data to analyze
        Y_test: data to compare the results
        
    
    Runs the model and prints the classification report
    """
    # predict using the model
    Y_pred = model.predict(X_test)
    
    # print the classification report
    for i in range(Y_pred.shape[1]):
        print(classification_report(np.array(Y_test)[i,:], Y_pred[i,:]))


def save_model(model, model_filepath):
    """
    Input:
        model: model to save in the pickle file
        model_filepath: path that indicates where to save the pickle file
        
    Saves the model object in a pickle file in the provided filepath.
    """
    pickle.dump(model, open(model_filepath, 'wb'))
    

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()