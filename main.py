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


