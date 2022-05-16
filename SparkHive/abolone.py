from os.path import join, abspath
from timeit import default_timer as timer
from pyspark.ml.feature import VectorAssembler, StringIndexer, Binarizer, MinMaxScaler
from pyspark.ml.classification import RandomForestClassifier, MultilayerPerceptronClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml import Pipeline
import json

def evaluate(spark, classifier, classifier_name):
    # Data Loading
    start_data_loading = timer()

    abolone = spark.sql("SELECT * FROM Abolone").cache()

    train, test = abolone.randomSplit([0.7, 0.3])

    end_data_loading = timer()

    data_loading_time = end_data_loading - start_data_loading

    # Training
    start_training = timer()

    sex_indexer = StringIndexer(inputCol='Sex', outputCol='SexIndexed')

    numeric_cols = [ 'SexIndexed', 'Length', 'Diameter', 'Height', 'WholeWeight', 'ShuckedWeight', 'VisceraWeight', 'ShellWeight' ]
    feature_indexer = VectorAssembler(inputCols=numeric_cols, outputCol="features")

    mm_scaler = MinMaxScaler(inputCol="features", outputCol="featuresMM")

    binarizer = Binarizer(threshold=10.0, inputCol="Rings", outputCol="label")

    pipeline = Pipeline(stages=[sex_indexer, feature_indexer, mm_scaler, binarizer, classifier])

    model = pipeline.fit(train)

    end_training = timer()

    training_time = end_training - start_training

    # Evaluation
    start_evaluation = timer()

    result = model.transform(test)

    accuracy_evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
    accuracy = accuracy_evaluator.evaluate(result)

    f1_evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="f1")
    f1 = f1_evaluator.evaluate(result)

    end_evaluation = timer()

    evaluation_time = end_evaluation - start_evaluation

    data = {
        "DataLoadingTime": data_loading_time,
        "TrainingTime": training_time,
        "EvaluationTime": evaluation_time,
        "Accuracy": accuracy,
        "F1Score": f1,
    }

    print(json.dumps(data, ensure_ascii=False, indent=4))

    with open(abspath(join("SparkHive", 'abolone_' + classifier_name + '.json')), 'w') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def evaluate_randomforest(spark):
    rf = RandomForestClassifier(featuresCol='featuresMM', labelCol='label', numTrees=300)
    evaluate(spark, rf, "randomforest")

def evaluate_multilayerperceptron(spark):
    layers = [8, 5, 4, 2]
    mlp = MultilayerPerceptronClassifier(featuresCol='featuresMM', labelCol='label', maxIter=100, layers=layers, blockSize=64)
    evaluate(spark, mlp, "multilayerperceptron")