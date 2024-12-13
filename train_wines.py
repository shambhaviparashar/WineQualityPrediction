from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml.feature import VectorAssembler, StringIndexer, MinMaxScaler, IndexToString
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml import Pipeline

spark = SparkSession.builder.appName("WineQualityPrediction").getOrCreate()

def clean_column_names(df):
    for col_name in df.columns:
        clean_name = col_name.replace('"', '').strip()
        df = df.withColumnRenamed(col_name, clean_name)
    return df

training_data_path = "s3://wine-quality-datasets/TrainingDataset.csv"
df = spark.read.csv(training_data_path, header=True, sep=';', inferSchema=True)

df = clean_column_names(df)

print("Class distribution in training data:")
df.groupBy("quality").count().show()

feature_cols = [
    'fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
    'chlorides', 'free sulfur dioxide', 'total sulfur dioxide',
    'density', 'pH', 'sulphates', 'alcohol'
]
target_col = 'quality'

df = df.dropna(subset=feature_cols + [target_col])

majority_class = df.filter(col("quality") == 5)
minority_classes = df.filter(col("quality") != 5)
balanced_df = majority_class.sample(fraction=0.5, seed=42).union(minority_classes)

assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")

scaler = MinMaxScaler(inputCol="features", outputCol="scaledFeatures")

indexer = StringIndexer(inputCol=target_col, outputCol="label")

label_converter = IndexToString(inputCol="prediction", outputCol="predicted_quality", labels=indexer.fit(df).labels)

rf = RandomForestClassifier(
    featuresCol="scaledFeatures", labelCol="label",
    numTrees=1000, maxDepth=30, minInstancesPerNode=2, impurity="entropy"
)

pipeline = Pipeline(stages=[assembler, scaler, indexer, rf, label_converter])

print("Training the model on balanced data in parallel...")
model = pipeline.fit(balanced_df)

model_save_path = "s3://wine-quality-datasets/models/wine_quality_model"
print(f"Saving the trained model to {model_save_path}...")
model.write().overwrite().save(model_save_path)

spark.stop()

print("Model training and saving completed.")
