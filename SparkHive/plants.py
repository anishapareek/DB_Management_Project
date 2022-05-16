from os.path import join, abspath
from timeit import default_timer as timer
from pyspark.ml.feature import VectorAssembler, StringIndexer, Binarizer, MinMaxScaler
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.ml import Pipeline
from pyspark.sql.types import StructType, StructField, DoubleType, FloatType, StringType, IntegerType
import json
import csv

def evaluate(spark, classifier, classifier_name):
    # Data Loading
    start_data_loading = timer()

    states = []

    with open(abspath(join("setup", "stateabbr.txt")), newline='') as csvfile:
        statereader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for row in statereader:
            states.append(row[0])

    plants = spark.sql("SELECT * FROM Plants").cache()

    train, test = plants.randomSplit([0.7, 0.3])

    end_data_loading = timer()

    data_loading_time = end_data_loading - start_data_loading

    # Training
    start_training = timer()

    feature_indexer = VectorAssembler(inputCols=states, outputCol="features")

    pipeline = Pipeline(stages=[feature_indexer, classifier])

    model = pipeline.fit(train)

    end_training = timer()

    training_time = end_training - start_training

    # Evaluation
    start_evaluation = timer()

    result = model.transform(test)

    silhouette_evaluator = ClusteringEvaluator( metricName="silhouette")
    silhouette = silhouette_evaluator.evaluate(result)

    end_evaluation = timer()

    evaluation_time = end_evaluation - start_evaluation

    data = {
        "DataLoadingTime": data_loading_time,
        "TrainingTime": training_time,
        "EvaluationTime": evaluation_time,
        "Silhouette": silhouette,
    }

    print(json.dumps(data, ensure_ascii=False, indent=4))

    with open(abspath(join("SparkHive", 'plants_' + classifier_name + '.json')), 'w') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def evaluate_kmeans(spark):
    kmeans = KMeans().setK(6)
    evaluate(spark, kmeans, "k-means")