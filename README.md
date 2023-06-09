# udacity-capstone-spark

This is a PySpark project for the udacity program. The aim of the project is to analyze activity of a music app users and train a model that would predict users that would cancel their subscription.
The project uses PySpark as a main tool. It follows several steps:

1. Load user data and clean it to remove records with empty or null user and session ids.
2. Identify sets of users that canceled their subscriptions and users that did not on the basis of page column.
    a. About quarter of two hundred distinct users decided to cancel subscription in the span of the dataset.
3. Perform behaviour analysis on various user activity features comparing averages between churned and subscrubing customers such as songs played per day, likes per song played, songs added to playlist per day etc. to identify what activity differs the most between the two groups.
5. Finally, separate user data into test and train sets and use the train data to train classification models (logistic regression and decision tree) to predict user churn. Having obtained the trained model, create predictions based on the test data and compare it against actual churn data.
5. Both logistic regression and decision tree show relatively high probability to predict the churn, f1 score for the regression is 0.7 and f1 score for the decision tree is 0.8.
6. The acccompanying analytical blog can be found here:
   https://medium.com/@iaroslav-miller/predicting-churn-among-music-app-users-2a2ee9cd0d8


List of libraries used:
1. pyspark
2. matplotlib
3. pandas
4. datetime

Description of file in the repository:
1. Sparkify.ipynb is a jupyter notebook of the project.
2. main.pu is a python file of the project.
3. readme.md is a readme file.

Acknowledgements

I am greatefull to Udacity and reviwers in particular for their help during this course and project.
