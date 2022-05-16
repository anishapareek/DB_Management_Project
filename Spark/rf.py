from pyspark.sql import SparkSession
import pandas as pd
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import StringIndexer
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import time
start1 = time.process_time()
spark = SparkSession.builder.appName('rf').getOrCreate()
df = spark.read.csv('abalone.csv', header=True, inferSchema=True)
print("Read Time: ",time.process_time() - start1)
pd.DataFrame(df.take(5), columns=df.columns).transpose()
numeric_features = [t[0] for t in df.dtypes if t[1] == 'double']
df.select(numeric_features).describe().toPandas().transpose()


numericCols = ['Length', 'Diameter', 'Height', 'WholeWeight', 'ShuckedWeight', 'VisceraWeight', 'ShellWeight', 'Rings']
assembler = VectorAssembler(inputCols=numericCols, outputCol="features")
df = assembler.transform(df)

label_stringIdx = StringIndexer(inputCol='Sex', outputCol='labelIndex')
df = label_stringIdx.fit(df).transform(df)
pd.DataFrame(df.take(110), columns=df.columns).transpose()
train, test = df.randomSplit([0.7, 0.3], seed=42)


start2 = time.process_time()
rf = RandomForestClassifier(featuresCol='features', labelCol='labelIndex')
rfModel = rf.fit(train)
print("Model Time: ",time.process_time() - start2)

start3 = time.process_time()
predictions = rfModel.transform(test)
predictions.select('Sex', 'Length', 'Diameter', 'Height', 'WholeWeight', 'ShuckedWeight', 'VisceraWeight',
                   'ShellWeight', 'Rings', 'labelIndex', 'rawPrediction', 'prediction', 'probability').show(25)
predictions.select("labelIndex", "prediction").show(10)

evaluator = MulticlassClassificationEvaluator(labelCol="labelIndex", predictionCol="prediction")
accuracy = evaluator.evaluate(predictions)
print("Accuracy = %s" % accuracy)
print("Test Error = %s" % (1.0 - accuracy))
print("Predict Time: ",time.process_time() - start3)
