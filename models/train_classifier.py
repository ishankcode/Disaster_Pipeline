import sys
import pandas as pd
import numpy as np
import re
import os

from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split, GridSearchCV 
import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk import pos_tag
import re
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import  f1_score,precision_score,recall_score,accuracy_score,make_scorer, classification_report
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from sklearn.base import BaseEstimator, TransformerMixin

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

def load_data(database_filepath):
    engine = create_engine('sqlite:///DisasterResponse.db')
    df = pd.read_sql_table('DisasterResponse_table', engine)
    X = df.message
    y = df.drop(['id', 'message', 'original', 'genre'], axis = 1)


def tokenize(text):
    """
    inputs:
    messages
       
    Returns:
    list of words into numbers of same meaning
    """
    # Normalize text (lowercase and remove all punctuations low level details dont matter())
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    pipeline = Pipeline([
               ('vect', CountVectorizer(tokenizer=tokenize)),
                    ('tfidf', TfidfTransformer()),
                    ('clf', MultiOutputClassifier(RandomForestClassifier()))
               ])
    parameters = {
              'tfidf__use_idf':[True, False],
              'clf__estimator__n_estimators':[50, 100], 
              'clf__estimator__min_samples_split':[2, 5]
             }
    cv = GridSearchCV(pipeline,param_grid=parameters, n_jobs=-1, verbose=2)
    
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    y_pred_test =model.predict(X_test)
    print(classification_report(Y_test.values, y_pred_test, target_names=y.columns.values))
    pass


def save_model(model, model_filepath):
    pickle.dump(model, open('final_model1.sav', 'wb'))


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
        evaluate_model(model, X_test, Y_test, category_names)

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