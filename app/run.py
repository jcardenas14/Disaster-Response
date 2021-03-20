import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from wordcloud import WordCloud

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

# load in nltk stopwords
stop_words = stopwords.words('english')

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

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('disaster_cleaned', engine)

# load trained model
model = joblib.load("../models/classifier.pkl")

# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    
    # DataFrame showing count frequency of message genres in training set
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    # DataFrame showing count frequency of message categories in training set
    categories = df.columns[4:]
    df_categories = df.melt(value_vars = categories).groupby('variable').sum()['value'].sort_values(ascending=False).reset_index()
    
    # Create word cloud for cleaned and processed training set
    messages_cleaned = [' '.join(tokenize(x)) for x in df.message]
    wordcloud = WordCloud(max_words=100).generate(' '.join(messages_cleaned))
    
    # Save wordcloud to png
    wordcloud.to_file("static/wordcloud.png")
    
    # create visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=df_categories.variable,
                    y=df_categories.value,
                    textposition='inside'
                )
            ],
            
            'layout': {
                'title': 'Distribution of Message Categories',
                'yaxis': {
                    'title': "Count",
                    'layer': "below traces"
                },
                'xaxis': {
                    'title': "",
                    'layer':"below traces"
                }
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(debug=True)


if __name__ == '__main__':
    main()