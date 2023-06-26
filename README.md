epository# Disaster Response Pipeline Project

## Project Description
In this project (Part of Udacity Data Science Nanodegree), I build a model to classify messages that are sent during disasters. There are 36 categories and during disasters, it is really important that messages are reported to the correct designated teams only (because during disasters there is an influx of messages). By classifying these messages properly, we can allow these messages to be sent to the appropriate disaster relief agency only. This project was in 3 main parts :
 - Building an ETL Pipeline- cleaning and transforming data and loading the data in the Database
 - Machine Learning pipeline- Built a Multi-model classification pipeline (the message can belong to one or more categories)
 - Built a Web app using HTML, CSS, JavaScript, and Python Flask.
 
## Dataset
For This Project, I will be working with a data set provided by [Figure Eight](https://www.figure-eight.com/) containing real messages that were sent during disaster events. (Can see the dataset in the repository)


![Disasters_gif](https://github.com/ishankcode/Disaster_Pipeline/assets/66678343/8e477f58-09fd-4a10-90f2-70711cfef703)



![app_1](https://github.com/ishankcode/Disaster_Pipeline/assets/66678343/fc986549-fba7-431a-a2bf-adfb16ce76af)
![app_2](https://github.com/ishankcode/Disaster_Pipeline/assets/66678343/51da44d9-91e9-49f6-acb5-8673fcda5261)


## File Description
~~~~~~~
        disaster_response_pipeline
          |-- app
                |-- templates
                        |-- go.html
                        |-- master.html
                |-- run.py
          |-- data
                |-- disaster_message.csv
                |-- disaster_categories.csv
                |-- DisasterResponse.db
                |-- process_data.py
          |-- models
                |-- classifier.pkl (Could not load to Guthub because of huge file - attaching a link to Googledrive )
                |-- train_classifier.py
          |-- Preparation
                |-- categories.csv
                |-- ETL Pipeline Preparation.ipynb
                |-- ETL_Preparation.db
                |-- messages.csv
                |-- ML Pipeline Preparation.ipynb
          |-- README
~~~~~~~

## Required Libraries
Machine Learning Libraries: Pandas, NumPy, Scikit-Learn
Natural Language Process Libraries: NLTK
SQLite Database Libraries: SQLalchemy
Web App libraries: Flask
Visualization libraries: Plotly, Matplotlib

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Go to `app` directory: `cd app`

3. Run your web app: `python run.py`

4. Click on the th link shown
