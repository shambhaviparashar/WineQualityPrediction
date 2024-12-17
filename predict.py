import sys
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when
from pyspark.ml import PipelineModel
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

if len(sys.argv) != 3:
    print("Usage: python predict.py <validation_dataset_path> <model_path>")
    sys.exit(1)

validation_dataset_path = sys.argv[1]
model_path = sys.argv[2]

# Initialize SparkSession
spark = SparkSession.builder \
    .appName("WineQualityPrediction") \
    .getOrCreate()

def clean_column_names(df):
    for c in df.columns:
        clean_name = c.strip().replace('"', '')
        df = df.withColumnRenamed(c, clean_name)
    return df

# Load validation data
df = spark.read.csv(validation_dataset_path, header=True, inferSchema=True, sep=';')
df = clean_column_names(df)

# Convert to binary label
df = df.withColumn("quality_label", when(col("quality") >= 7, 1).otherwise(0))

# Load the trained model
model = PipelineModel.load(model_path)

# Apply the model to the validation data
predictions = model.transform(df)

# Evaluate using F1 score
evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="f1")
f1_score = evaluator.evaluate(predictions)

# Print only the F1 score
print(f1_score)

spark.stop()
