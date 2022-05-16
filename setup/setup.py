from os.path import join, abspath
from pyspark.sql import SparkSession
from pyspark.sql.types import DoubleType, IntegerType, StructType, StructField, FloatType, StringType
import csv

warehouse_location = abspath('spark-warehouse')

spark = SparkSession \
    .builder \
    .appName("CS729 Project") \
    .config("spark.sql.warehouse.dir", warehouse_location) \
    .enableHiveSupport() \
    .getOrCreate()

spark.sql("CREATE TABLE IF NOT EXISTS Abolone (Sex Char(1), Length DOUBLE, Diameter DOUBLE, Height DOUBLE, WholeWeight DOUBLE, ShuckedWeight DOUBLE, VisceraWeight DOUBLE, ShellWeight DOUBLE, Rings DOUBLE) USING HIVE")

# Abolone
abolone_schema = StructType([
        StructField("Sex", StringType(), False),
        StructField("Length", FloatType(), False),
        StructField("Diameter", FloatType(), False),
        StructField("Height", FloatType(), False),
        StructField("WholeWeight", FloatType(), False),
        StructField("ShuckedWeight", FloatType(), False),
        StructField("VisceraWeight", FloatType(), False),
        StructField("ShellWeight", FloatType(), False),
        StructField("Rings", DoubleType(), False),
    ])
abolone = spark.read.csv(abspath(join("setup", "abalone.data")), header=False, schema=abolone_schema)
abolone.write.mode("overwrite").saveAsTable("Abolone")

abolone_test = spark.sql("SELECT * FROM Abolone LIMIT 5")
abolone_test.show()

states = []
plants_schema_fields = [ StructField("Nane", StringType(), False) ]

with open(abspath(join("setup", "stateabbr.txt")), newline='') as csvfile:
    statereader = csv.reader(csvfile, delimiter=' ', quotechar='|')
    for row in statereader:
        states.append(row[0] + " INT ")
        plants_schema_fields.append(StructField(row[0], IntegerType(), False))

plants_str = "CREATE TABLE IF NOT EXISTS Plants (Name String, " + ', '.join(states) + ") USING HIVE"
spark.sql(plants_str)

plants_schema = StructType(plants_schema_fields)
plants = spark.read.csv(abspath(join("setup", "plants.output")), header=False, schema=plants_schema)
plants.write.mode("overwrite").saveAsTable("Plants")

plants_test = spark.sql("SELECT * FROM Plants LIMIT 5")
plants_test.show()

facebook_schema = StructType([
    StructField("PagePopularity", FloatType(), False),
    StructField("PageCheckins", FloatType(), False),
    StructField("PageTalkingAbout", FloatType(), False),
    StructField("PageCategory", StringType(), False),
    StructField("Derived1", FloatType(), False),
    StructField("Derived2", FloatType(), False),
    StructField("Derived3", FloatType(), False),
    StructField("Derived4", FloatType(), False),
    StructField("Derived5", FloatType(), False),
    StructField("Derived6", FloatType(), False),
    StructField("Derived7", FloatType(), False),
    StructField("Derived8", FloatType(), False),
    StructField("Derived9", FloatType(), False),
    StructField("Derived10", FloatType(), False),
    StructField("Derived11", FloatType(), False),
    StructField("Derived12", FloatType(), False),
    StructField("Derived13", FloatType(), False),
    StructField("Derived14", FloatType(), False),
    StructField("Derived15", FloatType(), False),
    StructField("Derived16", FloatType(), False),
    StructField("Derived17", FloatType(), False),
    StructField("Derived18", FloatType(), False),
    StructField("Derived19", FloatType(), False),
    StructField("Derived20", FloatType(), False),
    StructField("Derived21", FloatType(), False),
    StructField("Derived22", FloatType(), False),
    StructField("Derived23", FloatType(), False),
    StructField("Derived24", FloatType(), False),
    StructField("Derived25", FloatType(), False),
    StructField("CC1", FloatType(), False),
    StructField("CC2", FloatType(), False),
    StructField("CC3", FloatType(), False),
    StructField("CC4", FloatType(), False),
    StructField("CC5", FloatType(), False),
    StructField("BaseTime", FloatType(), False),
    StructField("PostLength", FloatType(), False),
    StructField("PostShareCount", FloatType(), False),
    StructField("PostPromotionStatus", FloatType(), False),
    StructField("HLocal", FloatType(), False),
    StructField("PostPublishedWeekday1", FloatType(), False),
    StructField("PostPublishedWeekday2", FloatType(), False),
    StructField("PostPublishedWeekday3", FloatType(), False),
    StructField("PostPublishedWeekday4", FloatType(), False),
    StructField("PostPublishedWeekday5", FloatType(), False),
    StructField("PostPublishedWeekday6", FloatType(), False),
    StructField("PostPublishedWeekday7", FloatType(), False),
    StructField("BaseDateTimeWeekday1", FloatType(), False),
    StructField("BaseDateTimeWeekday2", FloatType(), False),
    StructField("BaseDateTimeWeekday3", FloatType(), False),
    StructField("BaseDateTimeWeekday4", FloatType(), False),
    StructField("BaseDateTimeWeekday5", FloatType(), False),
    StructField("BaseDateTimeWeekday6", FloatType(), False),
    StructField("BaseDateTimeWeekday7", FloatType(), False),
    StructField("TargetVariable", FloatType(), False)
])

fields = []
for field in facebook_schema.fields:
    if (field.dataType == FloatType()):
        field_type = "DOUBLE"
    elif (field.dataType == StringType()):
        field_type = "STRING"
    fields.append(field.name + " " + field_type)
facebook_str = "CREATE TABLE IF NOT EXISTS Plants (Name String, " + ', '.join(fields) + ") USING HIVE"
spark.sql(facebook_str)

facebook = spark.read.csv(abspath(join("setup", "facebook.csv")), header=False, schema=facebook_schema)
facebook.write.mode("overwrite").saveAsTable("Facebook")

plants_test = spark.sql("SELECT * FROM Facebook LIMIT 5")
plants_test.show()