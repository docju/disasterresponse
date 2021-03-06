import sys
# import libraries
import nltk
nltk.download(['punkt','wordnet','averaged_perceptron_tagger','stopwords'])

import re
import string
import numpy as np
import pandas as pd
import pickle

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sqlalchemy import create_engine
from sklearn.base import BaseEstimator
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
# import libraries
import nltk
nltk.download(['punkt','wordnet','averaged_perceptron_tagger','stopwords'])
from sklearn.neighbors import KDTree
from sklearn.model_selection import train_test_split


def load_data(database_filepath):
    '''
    INPUT:
    Location of file for analysis
    OUTPUT:
    X (input) Y (target) files
    
    '''
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table('MESSAGE_CATEGORIES', engine)
    print(df.head())
    X = df['message'].values
    Y = df[df.columns[4:]]
    cat_names=Y.columns
  

    return X,Y,cat_names


def tokenize(text):
    '''
    INPUT:
    X- containing messages
    OUTPUT:
    Tokenized and Lemmatized data without stopwords and punctuation
    
    '''
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls=re.findall(url_regex,text)
    for character in string.punctuation:
        text=text.replace(character,'')
    words=word_tokenize(text)
    tokens=[w for w in words if w not in stopwords.words ("english")]
    lemmatizer=WordNetLemmatizer()
    
    clean_tokens=[]
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
        
    return clean_tokens






def build_model():
    '''
    INPUT:
    Pipeline,X and Y tables
    OUTPUT:
    Built logistic regression model
    
    '''
    pipeline = Pipeline ([
              ('vect', CountVectorizer(tokenizer=tokenize)),
              ('tfidf',TfidfTransformer()),
               ('clf', MultiOutputClassifier(DecisionTreeClassifier())),
          ])
    parameters = [
                         {
 
            'vect__ngram_range': ((1, 1), (1, 2)),
            'clf__estimator__min_samples_leaf': [1, 10]
            },
            
            ]
    model = GridSearchCV(pipeline,param_grid=parameters,n_jobs=-1,cv=2)
    return model


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    INPUT:
    model, test datasets, category names
    OUTPUT:
    Classification report with precision, recall, f1 score
    
    '''
    
    y_pred=model.predict(X_test)

    accuracy=(y_pred==Y_test).mean()

    print(classification_report(Y_test, y_pred, target_names=category_names))
    print(accuracy)



def save_model(model, model_filepath):
    '''
    INPUT:
    model
    OUTPUT:
    Pickle model
    
    '''
    pickle.dump(model, open('classifier.pkl','wb'))


def main():
    '''
    Put all of the above together
    
    ''' 
   
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
