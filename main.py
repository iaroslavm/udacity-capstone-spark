# import libraries
import pyspark
from pyspark import SparkConf
from pyspark.sql import SparkSession
from pyspark.sql.functions import desc
from pyspark.sql.functions import udf
from pyspark.sql.types import StringType
from pyspark.sql.types import IntegerType
from pyspark.sql.types import LongType
# from pyspark.sql.functions import year, month, dayofmonth, hour
from pyspark.sql import Window
from pyspark.sql.functions import when, rand, mean, split, explode, max as maximum, min as minimum, datediff, \
    from_unixtime, \
    countDistinct as addupDistinct, sum as total, date_add

import datetime as dt

from pyspark.ml.classification import LogisticRegression, DecisionTreeClassifier, RandomForestClassifier
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

import matplotlib.pyplot as plt
import pandas as pd

# create timestamp udfs
get_hour = udf(lambda x: dt.datetime.fromtimestamp(x / 1000.0).hour)
get_year = udf(lambda x: dt.datetime.fromtimestamp(x / 1000.0).year)
get_month = udf(lambda x: dt.datetime.fromtimestamp(x / 1000.0).month)
get_day = udf(lambda x: dt.datetime.fromtimestamp(x / 1000.0).day)
# create data features udfs
check_for_churn = udf(lambda x: 1 if x == 'Cancellation Confirmation' else 0, IntegerType())
check_for_downgrade_visit = udf(lambda x: 1 if x == 'Downgrade' else 0, IntegerType())
check_for_downgrade_submit = udf(lambda x: 1 if x == 'Submit Downgrade' else 0, IntegerType())
check_for_playlist = udf(lambda x: 1 if x == 'Add to Playlist' else 0, IntegerType())
check_for_friends = udf(lambda x: 1 if x == 'Add Friend' else 0, IntegerType())
check_for_adverts = udf(lambda x: 1 if x == 'Roll Advert' else 0, IntegerType())
check_for_logout = udf(lambda x: 1 if x == 'Logout' else 0, IntegerType())
song_event = udf(lambda x: 1 if x == 'NextSong' else 0, IntegerType())
check_level = udf(lambda x: 1 if x == 'paid' else 0, IntegerType())
check_thumbsup = udf(lambda x: 1 if x == 'Thumbs Up' else 0, IntegerType())
check_thumbsdown = udf(lambda x: 1 if x == 'Thumbs Down' else 0, IntegerType())

# create window for partition
windowsum = Window.partitionBy("userId")


def get_new_spark_session():
    """
    Esteblish spark session
    :return: spark session
    """
    # create a Spark session
    spark = SparkSession \
        .builder \
        .appName("Sparkify app") \
        .getOrCreate()

    return spark


def load_data(spark):
    """
    Load data to analyze
    :param spark: existing spark session
    :return: loaded database
    """
    # load data
    path = "mini_sparkify_event_data.json"
    user_log = spark.read.json(path)

    return user_log


def clean_data(user_log):
    """
    Remove records with emptly or null userID or sessionId
    :param user_log: user activity database
    :return: cleaned user activity database
    """
    # filter out null and empty ids
    user_log = user_log.filter(user_log.userId != "") \
        .filter(user_log.userId.isNotNull()) \
        .filter(user_log.sessionId.isNotNull())

    # select particular columns to continue with
    selected_data = user_log.select(['sessionId', 'userId', 'sessionId',
                                     'song', 'artist', 'gender', 'itemInSession',
                                     'length', 'level', 'location', 'page', 'status',
                                     'ts', 'userAgent'])

    return selected_data


def create_time_columns(selected_data):
    """
    Create data features based on time dimension
    :param selected_data: cleaned user activity data
    :return: user activity data enriched with time dependent features
    """
    selected_data = selected_data.withColumn('year', get_year(selected_data.ts)) \
        .withColumn('month', get_month(selected_data.ts)) \
        .withColumn('day', get_day(selected_data.ts)) \
        .withColumn('hour', get_hour(selected_data.ts))

    # calculate start and end time for each user in the database
    selected_data = selected_data \
        .withColumn('endDate', from_unixtime(maximum(selected_data.ts / 1000.0).over(windowsum))) \
        .withColumn('startDate', from_unixtime(minimum(selected_data.ts / 1000.0).over(windowsum)))

    # calculate duration per user
    selected_data = selected_data \
        .withColumn('durationDays', datediff('endDate', 'startDate'))

    # adding weekly timestamps to track behaviour over time
    selected_data = selected_data.withColumn('weekBeforeEndDate', date_add(selected_data.endDate, -7)) \
        .withColumn('twoweekBeforeEndDate', date_add(selected_data.endDate, -14)) \
        .withColumn('threeweekBeforeEndDate', date_add(selected_data.endDate, -21)) \
        .withColumn('fourweekBeforeEndDate', date_add(selected_data.endDate, -28))

    return selected_data


def create_user_features(selected_data):
    """
    Enrich user activity data with user features describing user activity
    :param selected_data: user data
    :return: Enriched user data
    """
    selected_data = selected_data.withColumn('churn', check_for_churn('page')) \
        .withColumn('downgradeVisit', check_for_downgrade_visit('page')) \
        .withColumn('downgradeSubmit', check_for_downgrade_submit('page')) \
        .withColumn('addToPlaylist', check_for_playlist('page')) \
        .withColumn('addFriend', check_for_friends('page')) \
        .withColumn('rollAdvert', check_for_adverts('page')) \
        .withColumn('logout', check_for_logout('page')) \
        .withColumn('payingUser', check_level('level')) \
        .withColumn('Up', check_thumbsup('page')) \
        .withColumn('Down', check_thumbsdown('page')) \
        .withColumn('listeningSongs', song_event(selected_data.page))

    selected_data = selected_data.withColumn('churned', total('churn').over(windowsum))

    selected_data = selected_data.withColumn('paying_when_churned', when((selected_data.churn == 1)
                                                                         & (selected_data.payingUser == 1), 1)
                                             .otherwise(0))

    return selected_data


def calculate_total_user_activity(df):
    """
    Calculate aggregated user activity per user
    :param df: activity data
    :return: Data enriched with aggregated acitivty
    """
    # mark all instances of users that canceled subscription
    df = df.withColumn('downgradeVisited', total('downgradeVisit').over(windowsum)) \
        .withColumn('downgradeSubmitted', total('downgradeSubmit').over(windowsum))

    # calculate total songs listened per user
    df = df.withColumn('songsListened', total('listeningSongs').over(windowsum))
    df = df.withColumn('songsAddedToPlaylist', total('addToPlaylist').over(windowsum))
    df = df.withColumn('friendsAdded', total('addFriend').over(windowsum))
    df = df.withColumn('rolledAdverts', total('rollAdvert').over(windowsum))
    df = df.withColumn('logouts', total('logout').over(windowsum))
    df = df.withColumn('DownVoted', total('Down').over(windowsum))
    df = df.withColumn('UpVoted', total('Up').over(windowsum))

    return df


def calculate_days_of_actual_interaction(df):
    """
    Calculate daily involvement with the app by user
    :param df: User data
    :return: Enriched user data
    """
    days_interacting_per_user = df.select('userId', 'year', 'month', 'day').dropDuplicates() \
        .groupBy('userId').count().withColumnRenamed('count', 'daysInteracting')
    df = df.join(days_interacting_per_user, 'userId', 'left')

    return df


def calculate_user_activity_per_day(df):
    """
    Calculate user activity per day of involvement
    :param df: User data
    :return: User data enriched with daily activity
    """
    # calculate average amount of songs listened per user
    df = df.withColumn('songsListenedPerDay'
                       , df.songsListened / df.daysInteracting)
    # calculate average amount of songs listened per user
    df = df.withColumn('songsAddedToPlaylistPerDay'
                       , df.songsAddedToPlaylist / df.daysInteracting)
    # calculate average amount of songs listened per user
    df = df.withColumn('friendsAddedPerDay'
                       , df.friendsAdded / df.daysInteracting)
    # calculate average amount of songs listened per user
    df = df.withColumn('rolledAdvertsPerDay'
                       , df.rolledAdverts / df.daysInteracting)
    # calculate average amount of songs listened per user
    df = df.withColumn('logoutsPerDay'
                       , df.logouts / df.daysInteracting)
    df = df.withColumn('upVotedPerDay'
                       , df.UpVoted / df.daysInteracting)
    df = df.withColumn('DownVotedPerDay'
                       , df.DownVoted / df.daysInteracting)
    df = df.withColumn('DownVotedPerSong'
                       , df.DownVoted / df.songsListened)
    df = df.withColumn('UpVotedPerSong'
                       , df.UpVoted / df.songsListened)
    df = df.withColumn('rolledAdvertsPerSong'
                       , df.rolledAdverts / df.songsListened)

    return df


def prepare_data_for_analysis(selected_data_distinct_users):
    """
    Peprate user data for the classification analysis, replace nulls with zeros and transform for anlysis
    :param selected_data_distinct_users: data with distinct user aggregated behaviour
    :return: Transform user data
    """
    selected_data_distinct_users = selected_data_distinct_users.fillna(0, subset=['songsListenedPerDay'
        , 'DownVotedPerSong'
        , 'friendsAddedPerDay'
        , 'intensityOfInteraction'
        , 'playlistActivityChange'
        , 'listeningActivityChange'])

    data = selected_data_distinct_users.select(['churned'
                                                   , 'songsListenedPerDay'
                                                   , 'DownVotedPerSong'
                                                   , 'friendsAddedPerDay'
                                                   , 'intensityOfInteraction'
                                                   , 'playlistActivityChange'
                                                   , 'listeningActivityChange']).withColumnRenamed('churned', 'label')

    assembler = VectorAssembler(inputCols=["songsListenedPerDay"
        , "DownVotedPerSong"
        , 'friendsAddedPerDay'
        , 'intensityOfInteraction'
        , 'playlistActivityChange'
        , 'listeningActivityChange'], outputCol="features")

    data = assembler.transform(data)

    data = data.select(['label', 'features'])

    return data


def train_and_evaluate_model(train_data, test_data, classifier, evaluator):
    """
    Train model, evaluate its predicton against test data
    :param train_data: training data set
    :param test_data: testig data set
    :param classifier: classifier model
    :param evaluator: evaluating tool
    :return: trained model, classification predictions, f1 score
    """

    model = classifier.fit(train_data)

    predictions = model.transform(test_data)

    f1_score = evaluator.evaluate(predictions)

    return model, predictions, f1_score


def display_comparative_statistics(selected_data, selected_data_last_week, selected_data_before_last_week):
    """
    Display comparative statistics for chunred and remaining users
    :param selected_data: overall dataset
    :param selected_data_last_week: dataset for weekly activity for one week prior to the end of the sample
    :param selected_data_before_last_week: dataset for weekly activity for 2 weeks before the end of the sample
    """
    # print sample averages for selected variables
    selected_data_last_week.select(['userId', 'churned', 'songsListenedPerDay'
                                       , 'songsAddedToPlaylistPerDay'
                                       , 'friendsAddedPerDay'
                                       , 'rolledAdvertsPerDay'
                                       , 'logoutsPerDay'
                                       , 'upVotedPerDay'
                                       , 'DownVotedPerDay'
                                       , 'DownVotedPerSong'
                                       , 'UpVotedPerSong'
                                       , 'rolledAdvertsPerSong']) \
        .dropDuplicates() \
        .groupBy('churned') \
        .avg('songsListenedPerDay'
             , 'songsAddedToPlaylistPerDay'
             , 'friendsAddedPerDay'
             , 'rolledAdvertsPerDay').show()

    selected_data_before_last_week.select(['userId', 'churned', 'songsListenedPerDay'
                                              , 'songsAddedToPlaylistPerDay'
                                              , 'friendsAddedPerDay'
                                              , 'rolledAdvertsPerDay'
                                              , 'logoutsPerDay'
                                              , 'upVotedPerDay'
                                              , 'DownVotedPerDay'
                                              , 'DownVotedPerSong'
                                              , 'UpVotedPerSong'
                                              , 'rolledAdvertsPerSong']) \
        .dropDuplicates() \
        .groupBy('churned') \
        .avg('songsListenedPerDay'
             , 'songsAddedToPlaylistPerDay'
             , 'friendsAddedPerDay'
             , 'rolledAdvertsPerDay').show()

    # print averages for the last week of the sample
    selected_data_last_week.select(['userId', 'churned', 'songsListenedPerDay'
                                       , 'songsAddedToPlaylistPerDay'
                                       , 'friendsAddedPerDay'
                                       , 'rolledAdvertsPerDay'
                                       , 'logoutsPerDay'
                                       , 'upVotedPerDay'
                                       , 'DownVotedPerDay'
                                       , 'DownVotedPerSong'
                                       , 'UpVotedPerSong'
                                       , 'rolledAdvertsPerSong']) \
        .dropDuplicates() \
        .groupBy('churned') \
        .avg('songsListenedPerDay'
             , 'songsAddedToPlaylistPerDay'
             , 'friendsAddedPerDay'
             , 'rolledAdvertsPerDay').show()

    selected_data_last_week.select(['userId', 'churned', 'songsListenedPerDay'
                                       , 'songsAddedToPlaylistPerDay'
                                       , 'friendsAddedPerDay'
                                       , 'rolledAdvertsPerDay'
                                       , 'logoutsPerDay'
                                       , 'upVotedPerDay'
                                       , 'DownVotedPerDay'
                                       , 'DownVotedPerSong'
                                       , 'UpVotedPerSong'
                                       , 'rolledAdvertsPerSong']) \
        .dropDuplicates() \
        .groupBy('churned') \
        .avg('logoutsPerDay'
             , 'DownVotedPerSong'
             , 'UpVotedPerSong'
             , 'rolledAdvertsPerSong').show()

    # print data for averages two weeks before the end of the sample
    selected_data_before_last_week.select(['userId', 'churned', 'songsListenedPerDay'
                                              , 'songsAddedToPlaylistPerDay'
                                              , 'friendsAddedPerDay'
                                              , 'rolledAdvertsPerDay'
                                              , 'logoutsPerDay'
                                              , 'upVotedPerDay'
                                              , 'DownVotedPerDay'
                                              , 'DownVotedPerSong'
                                              , 'UpVotedPerSong'
                                              , 'rolledAdvertsPerSong']) \
        .dropDuplicates() \
        .groupBy('churned') \
        .avg('songsListenedPerDay'
             , 'songsAddedToPlaylistPerDay'
             , 'friendsAddedPerDay'
             , 'rolledAdvertsPerDay').show()

    selected_data_before_last_week.select(['userId', 'churned', 'songsListenedPerDay'
                                              , 'songsAddedToPlaylistPerDay'
                                              , 'friendsAddedPerDay'
                                              , 'rolledAdvertsPerDay'
                                              , 'logoutsPerDay'
                                              , 'upVotedPerDay'
                                              , 'DownVotedPerDay'
                                              , 'DownVotedPerSong'
                                              , 'UpVotedPerSong'
                                              , 'rolledAdvertsPerSong']) \
        .dropDuplicates() \
        .groupBy('churned') \
        .avg('logoutsPerDay'
             , 'DownVotedPerSong'
             , 'UpVotedPerSong'
             , 'rolledAdvertsPerSong').show()

    # print averaages for intensity of interaction
    selected_data.select(['userId', 'churned', 'intensityOfInteraction'])\
        .dropDuplicates() \
        .groupBy('churned') \
        .avg('intensityOfInteraction').show()


def plot_figures(activity_df):
    """
    Display figures to illustrate frequency distribution of page column values
    :param activity_df: pandas dataframe with page frequency to plot
    """
    activity_df.plot(x='page',y='count_percentage',kind='bar')
    plt.show()
    activity_df[activity_df['page'] != 'NextSong'].plot(x='page',y='count_percentage',kind='bar')
    plt.show()


def main():
    """
    Main driver to load, clean data, extract user activity features and train classification models
    """
    spark = get_new_spark_session()

    user_log = load_data(spark)

    activity_df = user_log.groupBy('page').count().orderBy(desc('count')).toPandas()
    activity_df['count_percentage'] = activity_df['count']/activity_df['count'].sum()*100
    plot_figures(activity_df)

    selected_data = clean_data(user_log)

    selected_data = create_time_columns(selected_data)

    selected_data = create_user_features(selected_data)

    # filter out last week of user interactions
    selected_data_last_week = selected_data \
        .filter(from_unixtime(selected_data.ts / 1000.0) > selected_data.weekBeforeEndDate)
    selected_data_before_last_week = selected_data \
        .filter(from_unixtime(selected_data.ts / 1000.0) < selected_data.weekBeforeEndDate)

    selected_data = calculate_total_user_activity(selected_data)
    selected_data_last_week = calculate_total_user_activity(selected_data_last_week)
    selected_data_before_last_week = calculate_total_user_activity(selected_data_before_last_week)

    selected_data = calculate_days_of_actual_interaction(selected_data)
    selected_data_last_week = calculate_days_of_actual_interaction(selected_data_last_week)
    selected_data_before_last_week = calculate_days_of_actual_interaction(selected_data_before_last_week)

    selected_data = calculate_user_activity_per_day(selected_data)
    selected_data_last_week = calculate_user_activity_per_day(selected_data_last_week)
    selected_data_before_last_week = calculate_user_activity_per_day(selected_data_before_last_week)

    selected_data = selected_data.withColumn('intensityOfInteraction',
                                             selected_data.daysInteracting / selected_data.durationDays * 100) \
        .fillna(0, subset=['intensityOfInteraction'])

    display_comparative_statistics(selected_data, selected_data_last_week, selected_data_before_last_week)

    # keep only distinct user calculations
    selected_data_last_week = selected_data_last_week.select(
        ['userId', 'songsListenedPerDay', 'songsAddedToPlaylistPerDay']).dropDuplicates()
    selected_data_before_last_week = selected_data_before_last_week.select(
        ['userId', 'songsListenedPerDay', 'songsAddedToPlaylistPerDay']).dropDuplicates()

    selected_data_last_week = selected_data_last_week \
        .withColumnRenamed('songsAddedToPlaylistPerDay', 'songsAddedToPlaylistPerDayLastWeek') \
        .withColumnRenamed('songsListenedPerDay', 'songsListenedPerDayLastWeek')

    selected_data_before_last_week = selected_data_before_last_week \
        .withColumnRenamed('songsAddedToPlaylistPerDay', 'songsAddedToPlaylistPerDayBeforeLastWeek') \
        .withColumnRenamed('songsListenedPerDay', 'songsListenedPerDayBeforeLastWeek')

    selected_data = selected_data.join(selected_data_last_week
                                       .select(['userId', 'songsAddedToPlaylistPerDayLastWeek'
                                                   , 'songsListenedPerDayLastWeek']), 'userId', 'left')

    selected_data = selected_data.join(selected_data_before_last_week
                                       .select(['userId', 'songsAddedToPlaylistPerDayBeforeLastWeek'
                                                   , 'songsListenedPerDayBeforeLastWeek']), 'userId', 'left')

    selected_data = selected_data.withColumn('playlistActivityChange',
                                             selected_data.songsAddedToPlaylistPerDayLastWeek /
                                             selected_data.songsAddedToPlaylistPerDayBeforeLastWeek) \
        .withColumn('listeningActivityChange',
                    selected_data.songsListenedPerDayLastWeek / selected_data.songsListenedPerDayBeforeLastWeek)

    selected_data_distinct_users = selected_data.select(['userId'
                                                            , 'churned'
                                                            , 'songsListenedPerDay'
                                                            , 'DownVotedPerSong'
                                                            , 'friendsAddedPerDay'
                                                            , 'intensityOfInteraction'
                                                            , 'playlistActivityChange'
                                                            , 'listeningActivityChange']).dropDuplicates()

    data = prepare_data_for_analysis(selected_data_distinct_users)

    # Split data into training and testing sets
    training_data, test_data = data.randomSplit([0.7, 0.3], seed=12345)

    # instanciate the linear regression model
    lr = LogisticRegression(maxIter=5, regParam=0.0)
    dtr = DecisionTreeClassifier(labelCol="label", featuresCol="features")

    # Define a grid of decision tree hyperparameters to search over
    paramGrid = ParamGridBuilder() \
        .addGrid(dtr.maxDepth, [4, 6]) \
        .build()

    # Evaluate the predictions using F1 score
    evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="f1")
    # Create a CrossValidator to perform grid search for decision tree
    cv = CrossValidator(estimator=dtr, estimatorParamMaps=paramGrid, evaluator=evaluator)

    # logistic regression
    lr_model, lr_predictions, lr_f1_score = train_and_evaluate_model(training_data, test_data, lr, evaluator)

    # decision tree
    dt_model, dt_predictions, dt_f1_score = train_and_evaluate_model(training_data, test_data, dtr, evaluator)

    # select best model from decision tree with greedsearch and calculate its f1 score
    dt_cvModel = cv.fit(training_data)
    best_model = dt_cvModel.bestModel
    best_model_predictions = best_model.transform(test_data)
    best_model_f1_score = evaluator.evaluate(best_model_predictions)

    # Create a dictionary with the column names and values
    scores = {'LR f1 score': [lr_f1_score],
         'DT default maxDepth of 5': [dt_f1_score],
         'DT maxDepth of 6': [best_model_f1_score]}

    # Create a DataFrame from the dictionary
    scores = pd.DataFrame(scores)

# Display the DataFrame
print(scores)

    if __name__ == '__main__':
        main()
