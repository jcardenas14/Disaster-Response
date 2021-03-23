# Disaster Response Pipeline

### Summary
This project consists of a web app that classifies input messages into zero, one, or multiple disaster related categories. The ability to accurately and automatedly classify messages during a disaster would help first responders and emergency relief organizations know where to focus their efforts to save more lives. A multi-output random forest classification model was trained on ~21k pre-labeled disaster related tweet and text messages provided by [Figure Eight](https://appen.com/). The process_data.py script runs the ETL pipeline that cleans and stores the data in DisasterResponse.db database. The train_classifier.py script runs the ML pipeline that trains the random forest classifier and saves the trained model in a pickle file, 'classifier.pkl'. 

### Files in the respository
**app** \
| - static \
| | - wordcloud.png # wordcloud of training data \
| - templates \
| | - master.html # overview page of web app \
| | - go.html # message classification results page \
| - run.py # Flask file that runs app \
**data** \
| - disaster_categories.csv # disaster category data to process \
| - disaster_messages.csv # disaster message data to process \
| - process_data.py # python script to process and combine data \
| - DisasterResponse.db # sql database where cleaned data is stored \
**models** \
| - train_classifier.py # script to train ML model \
| - classifier.pkl # saved ML model \
**README.md** # description of project

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://127.0.0.1:5000/
