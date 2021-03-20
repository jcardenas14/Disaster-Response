import numpy as np
import nltk
import re
import pandas as pd
import sys
import pickle
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score, precision_score, recall_score
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sqlalchemy import create_engine

# download nltk libraries and stopwords
nltk.download(['punkt', 'wordnet','stopwords','averaged_perceptron_tagger'])
stop_words = stopwords.words('english')

# function to load data
def load_data(database_filepath):
    '''
    load data from sql database given the database file path.
    
        Returns:
            X (DataFrame): DataFrame - each row is a message
            Y (DataFrame): DataFrame - each column is a category
            categories (list): List of category names
    '''
    
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table('disaster_cleaned', con=engine)
    X = df['message'].values
    Y = df.drop(columns = ['id', 'message', 'original', 'genre']).values
    categories = df.drop(columns = ['id', 'message', 'original', 'genre']).columns
    
    return X, Y, categories
    

def tokenize(text):
    """Returns list of processed and tokenized text given input text."""
    
    # tokenize text and convert to lower case
    tokens = [tok.lower() for tok in word_tokenize(text)]
    
    # remove stop words and non alpha-numeric characters
    tokens = [tok for tok in tokens if tok not in stop_words and tok.isalnum()]
    
    # initialize WordNetLemmatizer object
    lemmatizer = WordNetLemmatizer()

    # create list of lemmatized tokens
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    '''
    Returns multi-output random forest classifier pipeline.
    
    Construct pipeline for count vectorization of input text, TF-IDF 
    transformation, and initialization of multi-output
    random forest classifier. Initialize hyperparameter tuning 
    using GridSearchCV. 
    '''
    
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    parameters = {
        'clf__estimator__n_estimators': [50, 100, 200],
        'clf__estimator__min_samples_split': [2, 3, 4]
    }
    
    cv = GridSearchCV(pipeline, param_grid=parameters)
    
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    The transformers in the pipeline can be cached using ``memory`` argument.

    '''
    Returns f1 score, precision, and recall for each category.
    
    Parameters:
        model: trained model object
        X_test: DataFrame of test messages
        Y_test: DataFrame of test classified categories
        category_names: List of category names

    Returns:
        eval_df: DataFrame of f1 score, precision, and recall per category.
    '''
    
    # predict on test data
    y_pred = model.predict(X_test)
    
    # calculate f1 score, precision, and recall
    f1 = []
    precision = []
    recall = []
    for i in range(y_pred.shape[1]):
        f1.append(f1_score(Y_test[:,i], y_pred[:,i], average='macro', zero_division=0))
        precision.append(precision_score(Y_test[:,i], y_pred[:,i], average='macro', zero_division=0))
        recall.append(recall_score(Y_test[:,i], y_pred[:,i], average='macro'))
    eval_df = pd.DataFrame({"f1":f1, "precision":precision, "recall":recall}, index=category_names)
    
    return eval_df


def save_model(model, model_filepath):
    """Save trained model as pickle file to given path."""
    
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=21)
        
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