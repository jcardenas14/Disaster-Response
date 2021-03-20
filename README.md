# Disaster Response Pipeline

This project consists of a web app that classifies input messages into zero, one, or multiple disaster related categories. The multi-output random forest classification model was trained on ~21k pre-labeled disaster related tweet and text message data provided by [Figure Eight](https://appen.com/). The process_data.py script runs the ETL pipeline that cleans and stores the data in DisasterResponse.db database. The train_classifier.py script runs the ML pipeline that trains the random forest classifier and saves the trained model in a pickle file, 'classifier.pkl'. 

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://127.0.0.1:5000/
