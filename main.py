# import libraries
import pyspark
from pyspark import SparkConf
from pyspark.sql import SparkSession
from pyspark.sql.functions import desc
from pyspark.sql.functions import udf
from pyspark.sql.types import StringType
from pyspark.sql.types import IntegerType
from pyspark.sql.types import LongType
#from pyspark.sql.functions import year, month, dayofmonth, hour
from pyspark.sql import Window
from pyspark.sql.functions import rand, mean, split, explode, max as maximum,min as minimum,datediff,from_unixtime, countDistinct as addupDistinct, sum as total, date_add

import datetime as dt

from pyspark.ml.classification import LogisticRegression, DecisionTreeClassifier, RandomForestClassifier
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder


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
    # create a Spark session
    spark = SparkSession \
        .builder \
        .appName("Sparkify app") \
        .getOrCreate()

    return spark


def load_data(spark):
    # load data
    path = "mini_sparkify_event_data.json"
    user_log = spark.read.json(path)

    return user_log


def clean_data(user_log):
    # filter out null and empty ids
    user_log = user_log.filter(user_log.userId != "") \
        .filter(user_log.userId.isNotNull()) \
        .filter(user_log.sessionId.isNotNull())

    # select particular columns to continue with
    selected_data = user_log.select(['sessionId', 'userId', 'sessionId', \
                                     'song', 'artist', 'gender', 'itemInSession', \
                                     'length', 'level', 'location', 'page', 'status',\
                                     'ts', 'userAgent'])

    return selected_data


def create_time_columns(selected_data):
    selected_data = selected_data.withColumn('year', get_year(selected_data.ts)) \
        .withColumn('month', get_month(selected_data.ts)) \
        .withColumn('day', get_day(selected_data.ts)) \
        .withColumn('hour', get_hour(selected_data.ts))

    # calculate start and end time for each user in the database
    selected_data = selected_data\
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
    selected_data = selected_data.withColumn('churn', check_for_churn('page')) \
        .withColumn('downgradeVisit', check_for_downgrade_visit('page')) \
        .withColumn('downgradeSubmit', check_for_downgrade_submit('page')) \
        .withColumn('addToPlaylist', check_for_playlist('page')) \
        .withColumn('addFriend', check_for_friends('page')) \
        .withColumn('rollAdvert', check_for_adverts('page')) \
        .withColumn('logout', check_for_logout('page')) \
        .withColumn('paingUser', check_level('level')) \
        .withColumn('Up', check_thumbsup('page')) \
        .withColumn('Down', check_thumbsdown('page'))\
        .withColumn('listeningSongs', song_event(selected_data.page))

    selected_data = selected_data.withColumn('churned', total('churn').over(windowsum))

    return selected_data


def calculate_total_user_activity(df):

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



def main():
    spark = get_new_spark_session()

    user_log = load_data(spark)

    selected_data = clean_data(user_log)

    selected_data = create_time_columns(selected_data)

    selected_data = create_user_features(selected_data)

    # filter out last week of user interactions
    selected_data_last_week = selected_data \
        .filter(from_unixtime(selected_data.ts / 1000.0) > selected_data.weekBeforeEndDate)
    selected_data_before_last_week = selected_data \
        .filter(from_unixtime(selected_data.ts / 1000.0) < selected_data.weekBeforeEndDate)






