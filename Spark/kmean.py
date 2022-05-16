from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import StandardScaler
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
import matplotlib.pyplot as plt
import time

silhouette_score = []

start1 = time.process_time()
spark = SparkSession.builder.appName('kmean').getOrCreate()
data_customer = spark.read.csv('abalone.csv', header=True, inferSchema=True)
# data_customer.printSchema()
data_customer = data_customer.na.drop()
print("Read Time: ",time.process_time() - start1)

start2 = time.process_time()

numericCols = ['Length', 'Diameter', 'Height', 'WholeWeight', 'ShuckedWeight', 'VisceraWeight', 'ShellWeight', 'Rings']
assemble = VectorAssembler(inputCols=numericCols, outputCol="features")
assembled_data = assemble.transform(data_customer)
scale = StandardScaler(inputCol='features', outputCol='standardized')
data_scale = scale.fit(assembled_data)
data_scale_output = data_scale.transform(assembled_data)
print("Train Time: ",time.process_time() - start2)

start3 = time.process_time()
evaluator = ClusteringEvaluator(predictionCol='prediction', featuresCol='standardized',
                                metricName='silhouette', distanceMeasure='squaredEuclidean')
for i in range(2, 10):
    KMeans_algo = KMeans(featuresCol='standardized', k=i)
    KMeans_fit = KMeans_algo.fit(data_scale_output)
    output = KMeans_fit.transform(data_scale_output)
    score = evaluator.evaluate(output)
    silhouette_score.append(score)
    print("Silhouette Score:", score)
print("Train Time: ",time.process_time() - start3)



