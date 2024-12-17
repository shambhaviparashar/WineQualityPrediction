import sys
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when
from pyspark.ml import PipelineModel
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

if len(sys.argv) != 2:
    print("Usage: spark-submit validate.py <validation_dataset_path>")
    sys.exit(1)

validation_dataset_path = sys.argv[1]

model_path = "s3a://wine-quality-training/model2/wine_quality_model"
output_path = "s3a://wine-quality-training/output/predictions_validation"

spark = SparkSession.builder \
    .appName("WineQualityValidation") \
    .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem") \
    .config("spark.hadoop.fs.s3a.aws.credentials.provider", "com.amazonaws.auth.DefaultAWSCredentialsProviderChain") \
    .getOrCreate()

def clean_column_names(df):
    for c in df.columns:
        clean_name = c.strip().replace('"', '')
        df = df.withColumnRenamed(c, clean_name)
    return df

print(f"Loading validation dataset from {validation_dataset_path}...")
try:
    df = spark.read.csv(validation_dataset_path, header=True, inferSchema=True, sep=';')
    df = clean_column_names(df)
    print("Validation dataset loaded successfully. Preview:")
    df.show(5)
except Exception as e:
    print(f"Error loading validation dataset: {e}")
    sys.exit(1)

df = df.withColumn("quality_label", when(col("quality") >= 7, 1).otherwise(0))

print(f"Loading the trained model from {model_path}...")
try:
    model = PipelineModel.load(model_path)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    sys.exit(1)

print("Applying the model to the validation dataset...")
try:
    predictions = model.transform(df)
    print("Predictions generated successfully. Preview:")
    predictions.select("predicted_quality", "quality_label", "prediction").show(5)
except Exception as e:
    print(f"Error during prediction: {e}")
    sys.exit(1)

evaluator = MulticlassClassificationEvaluator(
    labelCol="label",
    predictionCol="prediction",
    metricName="f1"
)
f1_score = evaluator.evaluate(predictions)
print("F1 Score on Validation Set:", f1_score)

print(f"Saving predictions to {output_path}...")
try:
    predictions.select("predicted_quality", "quality_label", "prediction", "quality") \
               .write.mode("overwrite").csv(output_path)
    print("Predictions saved successfully.")
except Exception as e:
    print(f"Error saving predictions: {e}")
    sys.exit(1)

print("Validation process completed successfully.")
spark.stop()