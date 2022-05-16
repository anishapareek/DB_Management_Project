from os.path import join, abspath
from timeit import default_timer as timer
from pyspark.ml.feature import VectorAssembler, MinMaxScaler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml import Pipeline
import json

def evaluate(spark, classifier, classifier_name):
    # Data Loading
    start_data_loading = timer()

    facebook = spark.sql("SELECT * FROM Facebook").cache()

    train, test = facebook.randomSplit([0.7, 0.3])

    end_data_loading = timer()

    data_loading_time = end_data_loading - start_data_loading

    # Training
    start_training = timer()

    features_names = [
        "PagePopularity", 
        "PageCheckins", 
        "PageTalkingAbout", 
        "Derived1", 
        "Derived2", 
        "Derived3", 
        "Derived4", 
        "Derived5", 
        "Derived6", 
        "Derived7", 
        "Derived8", 
        "Derived9", 
        "Derived10", 
        "Derived11", 
        "Derived12", 
        "Derived13", 
        "Derived14", 
        "Derived15", 
        "Derived16", 
        "Derived17", 
        "Derived18", 
        "Derived19", 
        "Derived20", 
        "Derived21", 
        "Derived22", 
        "Derived23", 
        "Derived24", 
        "Derived25", 
        "CC1", 
        "CC2", 
        "CC3", 
        "CC4", 
        "CC5", 
        "BaseTime", 
        "PostLength", 
        "PostShareCount", 
        "PostPromotionStatus", 
        "HLocal", 
        "PostPublishedWeekday1", 
        "PostPublishedWeekday2", 
        "PostPublishedWeekday3", 
        "PostPublishedWeekday4", 
        "PostPublishedWeekday5", 
        "PostPublishedWeekday6", 
        "PostPublishedWeekday7", 
        "BaseDateTimeWeekday1", 
        "BaseDateTimeWeekday2", 
        "BaseDateTimeWeekday3", 
        "BaseDateTimeWeekday4", 
        "BaseDateTimeWeekday5", 
        "BaseDateTimeWeekday6", 
        "BaseDateTimeWeekday7"
    ]
    feature_indexer = VectorAssembler(inputCols=features_names, outputCol="features")

    mm_scaler = MinMaxScaler(inputCol="features", outputCol="featuresMM")

    pipeline = Pipeline(stages=[feature_indexer, mm_scaler, classifier])

    model = pipeline.fit(train)

    end_training = timer()

    training_time = end_training - start_training

    # Evaluation
    start_evaluation = timer()

    result = model.transform(test)

    mse_evaluator = RegressionEvaluator(predictionCol="prediction", labelCol="TargetVariable",metricName="mse")
    mse = mse_evaluator.evaluate(result)

    mae_evaluator = RegressionEvaluator(predictionCol="prediction", labelCol="TargetVariable",metricName="mae")
    mae = mae_evaluator.evaluate(result)

    r2_evaluator = RegressionEvaluator(predictionCol="prediction", labelCol="TargetVariable",metricName="r2")
    r2 = r2_evaluator.evaluate(result)

    rmse_evaluator = RegressionEvaluator(predictionCol="prediction", labelCol="TargetVariable",metricName="rmse")
    rmse = rmse_evaluator.evaluate(result)

    end_evaluation = timer()

    evaluation_time = end_evaluation - start_evaluation

    data = {
        "DataLoadingTime": data_loading_time,
        "TrainingTime": training_time,
        "EvaluationTime": evaluation_time,
        "MeanSquaredError": mse,
        "MeanAbsoluteError": mae,
        "RSquared": r2,
        "RootMeanSquaredError": rmse
    }

    print(json.dumps(data, ensure_ascii=False, indent=4))

    with open(abspath(join("SparkHive", 'facebook_' + classifier_name + '.json')), 'w') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def evaluate_linearregression(spark):
    lr = LinearRegression(regParam=0.0, solver="normal", featuresCol="featuresMM",  labelCol="TargetVariable")
    evaluate(spark, lr, "linearregression")