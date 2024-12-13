
# Wine Quality Prediction using Apache Spark and AWS EMR

This project implements a machine learning pipeline to predict wine quality scores based on various chemical properties of wine. The pipeline is trained using Apache Spark's MLlib on AWS EMR in a distributed environment, and the trained model is deployed using Docker for predictions.

---

## Project Overview

- **Objective:** 
  - Train a Random Forest classifier to predict wine quality based on chemical properties.
  - Deploy the trained model for predictions using a Dockerized application.
- **Environment:** 
  - Distributed training on AWS EMR with Spark.
  - Dockerized prediction application for scalability and portability.
- **Datasets:** 
  - Training: `TrainingDataset.csv`
  - Validation: `ValidationDataset.csv`
  - Testing: `TestDataset.csv`
- **Output:** 
  - Trained model saved in S3.
  - Predictions from a Docker container for testing datasets.

---

## Features

### 1. **Training Pipeline**
- Preprocessing: Cleans dataset, balances class distribution, scales features.
- Model: Random Forest classifier with 1000 trees and max depth of 30.
- Output: Trained model saved to S3 at `s3://wine-quality-datasets/models/wine_quality_model`.

### 2. **Prediction Application**
- Inputs a test dataset (`TestDataset.csv`) and outputs predicted wine quality scores along with model performance (F1 score).
- Dockerized for easy deployment and execution.

---

## Repository Structure

- `train_wines.py`: Script for training the model on AWS EMR.
- `predict_wines.py`: Script for running predictions on a test dataset using the trained model.
- `Dockerfile`: Docker configuration for the prediction application.
- `README.md`: Project documentation.

---

## Steps to Run the Project

### Prerequisites
1. **AWS Setup:**
   - EMR cluster with Spark installed.
   - S3 bucket for storing datasets and model artifacts.

2. **Tools Installed:**
   - Apache Spark
   - Docker
   - Python 3.x

---

### **1. Model Training on AWS EMR**
1. Upload the training dataset (`TrainingDataset.csv`) to S3:
   ```plaintext
   s3://wine-quality-datasets/TrainingDataset.csv
   ```

2. SSH into the EMR primary node:
   ```bash
   ssh -i my-emr-keypair.pem hadoop@<EMR_PRIMARY_NODE_DNS>
   ```

3. Run the training script:
   ```bash
   spark-submit train_wines.py
   ```

4. The trained model will be saved to:
   ```plaintext
   s3://wine-quality-datasets/models/wine_quality_model
   ```

---

### **2. Prediction Using Docker**

#### Dockerized Prediction Application
1. **Build the Docker Image:**
   - Use the `predict_wines.py` script to predict wine quality.
   - Build the Docker container:
     ```bash
     docker build -t wine-quality-prediction .
     ```

2. **Run the Docker Container:**
   - Use the test dataset (`TestDataset.csv`) as input:
     ```bash
     docker run -v $(pwd):/app wine-quality-prediction <path_to_test_file>
     ```

3. **Output:**
   - The predicted wine quality scores and the F1 score will be displayed in the terminal.

---

## Prediction Script (`predict_wines.py`)

The `predict_wines.py` script:
1. Loads the trained model from S3.
2. Reads the test dataset from the local file system.
3. Performs predictions and outputs:
   - Predicted wine quality scores.
   - F1 score of the model on the test dataset.

### Running the Script Manually
To run the prediction script locally (without Docker):
```bash
python predict_wines.py --model-path s3://wine-quality-datasets/models/wine_quality_model --test-file <path_to_test_file>
```

---

## Monitoring and Debugging

- **AWS EMR Logs:** Check logs archived in S3 for troubleshooting:
  ```plaintext
  s3://wine-quality-datasets/logs/
  ```
- **Spark History Server:** Monitor training progress via the EMR dashboard.
- **Docker Logs:** Use `docker logs` to debug container execution.

---

## Tools and Technologies
- **Apache Spark MLlib:** For scalable machine learning.
- **AWS EMR:** For distributed training.
- **Docker:** For containerizing the prediction application.
- **Python 3.x:** For pipeline and model implementation.

---

## Author
Shambhavi Parashar  
GitHub: [shambhaviparashar](https://github.com/shambhaviparashar/WineQualityPrediction)
