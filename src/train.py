"""
This module contains functions to preprocess and train the model
for bank consumer churn prediction.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder,  StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay)
from sklearn.ensemble import RandomForestClassifier

### Import MLflow
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from joblib import dump


def rebalance(data):
    """
    Resample data to keep balance between target classes.

    The function uses the resample function to downsample the majority class to match the minority class.

    Args:
        data (pd.DataFrame): DataFrame

    Returns:
        pd.DataFrame): balanced DataFrame
    """
    churn_0 = data[data["Exited"] == 0]
    churn_1 = data[data["Exited"] == 1]
    if len(churn_0) > len(churn_1):
        churn_maj = churn_0
        churn_min = churn_1
    else:
        churn_maj = churn_1
        churn_min = churn_0
    churn_maj_downsample = resample(
        churn_maj, n_samples=len(churn_min), replace=False, random_state=1234
    )

    return pd.concat([churn_maj_downsample, churn_min])


def preprocess(df):
    """
    Preprocess and split data into training and test sets.

    Args:
        df (pd.DataFrame): DataFrame with features and target variables

    Returns:
        ColumnTransformer: ColumnTransformer with scalers and encoders
        pd.DataFrame: training set with transformed features
        pd.DataFrame: test set with transformed features
        pd.Series: training set target
        pd.Series: test set target
    """
    filter_feat = [
        "CreditScore",
        "Geography",
        "Gender",
        "Age",
        "Tenure",
        "Balance",
        "NumOfProducts",
        "HasCrCard",
        "IsActiveMember",
        "EstimatedSalary",
        "Exited",
    ]
    cat_cols = ["Geography", "Gender"]
    num_cols = [
        "CreditScore",
        "Age",
        "Tenure",
        "Balance",
        "NumOfProducts",
        "HasCrCard",
        "IsActiveMember",
        "EstimatedSalary",
    ]
    data = df.loc[:, filter_feat]
    data_bal = rebalance(data=data)
    X = data_bal.drop("Exited", axis=1)
    y = data_bal["Exited"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=1912
    )
    col_transf = make_column_transformer(
        (StandardScaler(), num_cols), 
        (OneHotEncoder(handle_unknown="ignore", drop="first"), cat_cols),
        remainder="passthrough",
    )

    X_train = col_transf.fit_transform(X_train)
    X_train = pd.DataFrame(X_train, columns=col_transf.get_feature_names_out())

    X_test = col_transf.transform(X_test)
    X_test = pd.DataFrame(X_test, columns=col_transf.get_feature_names_out())

    # Log the transformer as an artifact
    mlflow.sklearn.log_model(
        sk_model=col_transf,
        artifact_path="column_transformer",
        registered_model_name="ColumnTransformerChurn",
        input_example=X_train.iloc[0:1],
    )

    return col_transf, X_train, X_test, y_train, y_test


def train(X_train, y_train):
    """
    Train a logistic regression model.

    Args:
        X_train (pd.DataFrame): DataFrame with features
        y_train (pd.Series): Series with target

    Returns:
        RandomForestClassifier: trained random forest classifier model
    """
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)

    ### Log the model with the input and output schema
    # Infer signature (input and output schema)

    signature = infer_signature(X_train, rf.predict(X_train))
    mlflow.sklearn.log_model(
        sk_model=rf,
        artifact_path="model",
        registered_model_name="RandomForestChurn",
        input_example=X_train.iloc[0:1],
        signature=signature,
    )

    ### Log the data
    mlflow.log_artifact("data/Churn_Modelling.csv")
    
    os.makedirs("model", exist_ok=True)
    ### Log the model
    dump(rf, "model/model.pkl")
    mlflow.log_artifact("model/model.pkl")
    

    return rf

# conda activate [PATH_TO_ENV]
# conda activate ./.churn_prediction

# To run the mlflow server:
# mlflow ui
# keep it running, to close it, use Ctrl+C

# run any python file: python [PATH_TO_FILE]
# run this file: python src/train.py

def main():
    ### Set the tracking URI for MLflow
    print("Starting Mlflow ui")
    mlflow.set_tracking_uri("http://localhost:5000")

    ### Set the experiment name
    print("Starting Experiment")
    mlflow.set_experiment("churn_prediction")

    ### Start a new run and leave all the main function code as part of the experiment
    print("Starting run")
    with mlflow.start_run():

        print("loading data")
        df = pd.read_csv("data/Churn_Modelling.csv")
        col_transf, X_train, X_test, y_train, y_test = preprocess(df)

        print("start logging")
        ### Log the max_iter parameter
        mlflow.log_param("max_iter", 1000)
        ### Log the column transformer as an artifact
        dump(col_transf, "column_transformer.pkl")
        mlflow.log_artifact("column_transformer.pkl")
        ### Log the data
        mlflow.log_artifact("data/Churn_Modelling.csv")

        print("start training")
        model = train(X_train, y_train)

    
        print("start predicting")
        y_pred = model.predict(X_test)
        
        print("logging metrics")
        ### Log metrics after calculating them
        mlflow.log_metric("accuracy", accuracy_score(y_test, y_pred))
        mlflow.log_metric("precision", precision_score(y_test, y_pred))
        mlflow.log_metric("recall", recall_score(y_test, y_pred))
        mlflow.log_metric("f1_score", f1_score(y_test, y_pred))


        ### Log tags
        print("Start logging tags")
        mlflow.set_tags({"model": "RandomForestClassifier", "dataset": "Churn_Modelling"})

    
        print("Print Confusion Matrix")
        conf_mat = confusion_matrix(y_test, y_pred, labels=model.classes_)
        conf_mat_disp = ConfusionMatrixDisplay(
            confusion_matrix=conf_mat, display_labels=model.classes_
        )

        print("plotting Confusion Matrix")
        conf_mat_disp.plot()

        print("log the image info artifact")
        # Log the image as an artifact in MLflow
        conf_mat_disp.figure_.savefig("confusion_matrix.png")
        mlflow.log_artifact("confusion_matrix.png")
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("True")
    
        plt.show()


if __name__ == "__main__":
    main()
