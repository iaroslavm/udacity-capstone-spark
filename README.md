# udacity-capstone-spark

This is a PySpark project for the udacity program. The aim of the project is to analyze acitivity of a music app users and train a model that would predict users that would cancel their subscription.
The project uses PySpark as a main tool. It follows several steps:

1. Load user data and clean it to remove records with empty or null user ids.
2. Identify users sets of users that canceled their subscriptions and users that did not.
3. Perform behaviour analysis on various user activity features comparing averages between churned and subscrubing customers such as songs played per day, likes per song played, songs added to playlist per day etc. to identify what activity differs the most between the two groups.
4. Finally, separate user data into test and train sets and use the train data to train classification models (logistic regression and decision tree) to predict user churn. Having obtained the trained model, create predictions based on the test data and compare it against actual churn data.
5. Both logistic regressiona nd decission tree show relatively high probability to predict the churn, f1 score for the regression is 0.7 and f1 score for the decision tree is 0.8. 
