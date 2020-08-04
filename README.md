# Disaster Response Pipeline Project

### Summary

This project extracts data from two datasets, one contains disaster messages while the other one contains disaster categories. Then this data is transformed so it is easier to analyze and load into a database. The database is used to train a model that will be able to predict the categories of the messages.

There is also a web app that displays visualizations of the data. Additionally, the web app can be used to input a message that the model will analyze to predict its categories and show the classification results. 

### Files

app/
	run.py: script that will run the Flask web app.

data/
	DisasterResponse.db: database created by the ETL script.
	disaster_categories.csv: dataset containing the data of the disaster categories.
	disaster_messages.csv: dataset containing the data of the disaster messages.
	process_data.py: ETL script.

models/
	train_classifier.py: this script will create and train the model.


### Instructions:

1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/


### Web App Screenshots

![Web App Screenshot 1](/screenshots/webapp_screenshot_1.png?raw=true "Optional Title")

![Web App Screenshot 2](/screeshots/webapp_screenshot_2.png?raw=true "Optional Title")