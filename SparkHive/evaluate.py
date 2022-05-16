from os.path import abspath
from pyspark.sql import SparkSession
from abolone import evaluate_randomforest, evaluate_multilayerperceptron
from plants import evaluate_kmeans
from facebook import evaluate_linearregression

if __name__ == "__main__":
    warehouse_location = abspath('spark-warehouse')

    spark = SparkSession \
        .builder \
        .appName("CS729 Project") \
        .config("spark.sql.warehouse.dir", warehouse_location) \
        .enableHiveSupport() \
        .getOrCreate()

    evaluate_randomforest(spark)
    evaluate_multilayerperceptron(spark)
    evaluate_kmeans(spark)
    evaluate_linearregression(spark)