import json
import plotly
import pandas as pd
import nltk
import re
import string


from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from plotly.graph_objs import Histogram
from sklearn.externals import joblib
from sqlalchemy import create_engine
from pandas import DataFrame


app = Flask(__name__)

def tokenize(text):
    """
    INPUT:
    X- containing messages
    OUTPUT:
    Tokenized and Lemmatized data without stopwords and punctuation
    
    """
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

# load data
engine = create_engine('sqlite:///STOTAJO.db')
df = pd.read_sql_table('MESSAGE_CATEGORIES', engine)

# load model
model = joblib.load('./models/classifier.pkl')


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # Count Genres (this part provided by Udacity)
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    topic_counts=[]
    topic_names=list(df.columns[4:])
    
    #Count the total number of messages assigned to each topic in the data
    for i in topic_names:
        col_sum=df[df[i]==1].count()['message']
        topic_counts.append(col_sum)
    #Count the total number of topics hit by each message
    df['total_topics']=0
    for j in topic_names:
       df['total_topics']=df['total_topics']+df[j]
    
    total_topics=list(df['total_topics'].unique()).sort()
    topic_numbers=df.groupby(['total_topics']).count()['message']

    # create visuals
    # Count genres (provided by Udacity)
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
                    'title': "Number of Messages"
                },
                'xaxis': {
                    'title': "Number of Topics"
                }
            }
        },
    # Count messages per topic
        {
            'data': [
                Bar(
                    x=topic_names,
                    y=topic_counts
                )
            ],

            'layout': {
                'title': 'Topic',
                'yaxis': {
                    'title': "Number of Messages"
                },
                'xaxis': {
                    'title': "Type"
                }
            }
        },
      # Count topics per message and the number of messages assigned to that number of topics
        {
            'data': [
              #  Histogram(
              #      x=total_topics
                Bar(x=total_topics,
                    y=topic_numbers
                )
            ],
  

            'layout': {
                
                'title': 'Number of topics per message',
                'yaxis': {
                    'title': "Number of Messages"
                },
                'xaxis': {
                    'title': "Number of Topics per Message"
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
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()
