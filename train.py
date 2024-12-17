from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when
from pyspark.ml.feature import VectorAssembler, StringIndexer, IndexToString
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml import Pipeline

#setting s3 cred for fecilitating the access to s3 bucket
spark = SparkSession.builder \
    .appName("WineQualityPrediction") \
    .config("spark.hadoop.fs.s3a.aws.credentials.provider", "org.apache.hadoop.fs.s3a.DefaultS3ClientFactory") \
    .getOrCreate()

#cleaning column names
def clean_column_names(df):
    for c in df.columns:
        clean_name = c.replace('"', '').strip()
        df = df.withColumnRenamed(c, clean_name)
    return df

#S3 path for input data and the model save location
training_data_path = "s3://wine-quality-training/input/TrainingDataset.csv"
model_save_path = "s3://wine-quality-training/model2/wine_quality_model"

#header=True: Treats the first row as column names. sep=';': Uses semicolon as a separator. inferSchema=True: Automatically detects column data types.
df = spark.read.csv(training_data_path, header=True, sep=';', inferSchema=True)
df = clean_column_names(df)

#data exploration
print("Initial Class distribution in data:")
df.groupBy("quality").count().show()

#input features to train the model
feature_cols = [
    'fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
    'chlorides', 'free sulfur dioxide', 'total sulfur dioxide',
    'density', 'pH', 'sulphates', 'alcohol'
]

#adding column quality_label based on the quality column
df = df.withColumn("quality_label", when(col("quality") >= 7, 1).otherwise(0))

#removes rows with null values
df = df.dropna(subset=feature_cols + ["quality_label"])

#data exploration after adding quality_label
print("Binary Class distribution (quality_label):")
df.groupBy("quality_label").count().show()

#VectorAssembler: A feature transformer that merges multiple columns into a vector column.
#feature_cols into a single column features
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")

#StringIndexer: A label indexer that maps a string column of labels to an ML column of label indices.
indexer = StringIndexer(inputCol="quality_label", outputCol="label")
#IndexToString: A label converter that maps a column of label indices back to a column containing the original labels as strings.
label_converter = IndexToString(inputCol="prediction", outputCol="predicted_quality", labels=["0", "1"])

#RandomForestClassifier: A random forest classifier that fits multiple decision trees and uses the average to improve the predictive accuracy \
    # and control over-fitting.
rf = RandomForestClassifier(
    featuresCol="features", 
    labelCol="label",
    numTrees=100, 
    maxDepth=10, 
    minInstancesPerNode=2, 
    impurity="entropy",
    seed=42
)

#
pipeline = Pipeline(stages=[assembler, indexer, rf, label_converter])

#Trains the pipeline (feature transformations + Random Forest model) on the DataFrame df
print("Training the model on the entire dataset...")
model = pipeline.fit(df)

print(f"Saving the trained model to {model_save_path}...")
model.write().overwrite().save(model_save_path)

spark.stop()

print("Model training and saving completed.")
