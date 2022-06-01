
### Table of Contents

1. [Installation](#installation)
2. [Project Motivation](#motivation)
3. [Process](#process)
4. [File Descriptions](#files)
5. [Processing](#processing)
6. [Results](#results)
7. [Licensing, Authors, and Acknowledgements](#licensing)

## Installation <a name="installation"></a>
A number of libraries need to be imported (this will be included in the Anaconda version of Python):
sys
nltk
re
string
numpy
pandas
pickle
sklearn
flask
json
plotly
sqlalchemy


## Project Motivation<a name="motivation"></a>

We want to understand how we can classify messages received in a disaster response scenario  so that appropriate help can be sent to the sender, and find the best model to do so. Along the way, we use different methods and select the best one to find the optimal model to classify from data we have received from Figure8/Appen. We will also provide some visualisations to better understand the distribution of the classifications as well as an API to perform this automatically.

The visualisations included are a count of the genres, the count of the number of messages that fit in to a given topic, and the total number of messages that have the given number of topics.

## Process <a name= "process"></a>
We used Jupyter notebooks to perform exploratory data analysis on the data, after which we created the ETL process in process_data.py. 
Using pipelines, we used various methods including logistic regression, random forests, and decision trees to determine the best model to use for the model. Once this was done, the refactored code was put into an IDE so it could be run from a terminal. 
Plotly visuals were created and added to the html documents, and then the flask app was created using a template provided by Udacity.


## File Descriptions <a name="files"></a>

The "app" folder contains a folder with the two html templates go.html and master.html which contain the resulting output from classifying the webpage and the visualisations respectively.

It also contains the main "run.py" code that runs the app.

The "data" folder contains the two raw csvs and the "process_data.py" python code which loads the data.

The "models" folder contains the python code that trains the classifier as well as the pickle file that is output from the classifier

## Processing<a name="processing"></a>

In order to run the code, navigate the correct workplace in the terminal.

Run process_data.py to load the data to the correct location

Run train_classifier.py to process the data and build the model (this takes a while)

Run env|grep WORK to find the domain and ID- the website you'll need to navigate to. It will be https://[ID]-3001.[domain]

Run app/run.py to start the app

Navigate to the webpage using the id and domain and input a message into the box, and it will return the classification.

## Results<a name=results"></a>
The final model is functional, but could perform better- one issue is that the "child_alone" category has no positive hits and so it is impossible to model on as it stands. All of the other categories have a good accuracy above 75% but some words such as "freezing" do not get assigned to obviously correct topics.
  
The visualisations show that "related" is a popular topic but it is not clear what that means. Otherwise, they show that messages can be classified in to more than one topic, and this is quite common.

## Licensing, Authors, Acknowledgements<a name="licensing"></a>

Data was provided by Figure8/Appen and the projects and templates were provided by Udacity.


