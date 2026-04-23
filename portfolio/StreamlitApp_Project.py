import os, sys, warnings
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import posixpath

import joblib
import tarfile
import tempfile

import boto3
import sagemaker
from sagemaker.predictor import Predictor
from sagemaker.serializers import JSONSerializer 
from sagemaker.deserializers import JSONDeserializer # FIXED: Match inference script
from sklearn.pipeline import Pipeline
import shap
from joblib import dump, load

# Setup & Path Configuration
warnings.simplefilter("ignore")

# Fix path for Streamlit Cloud (ensure 'src' is findable)
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

# NOTE: Ensure these exactly match the classes used in your finalized_fraud_model.pkl!
from src.Custom_Classes import DropHighMissingCols, TransactionFeatureEngineer, DropHighCorrelation

file_path = os.path.join(project_root, 'Portfolio/X_train.csv')
dataset = pd.read_csv(file_path)
dataset = dataset.loc[:, ~dataset.columns.str.contains('^Unnamed')]

# Access the secrets
aws_id = st.secrets["aws_credentials"]["AWS_ACCESS_KEY_ID"]
aws_secret = st.secrets["aws_credentials"]["AWS_SECRET_ACCESS_KEY"]
aws_token = st.secrets["aws_credentials"]["AWS_SESSION_TOKEN"]
aws_bucket = st.secrets["aws_credentials"]["AWS_BUCKET"]
aws_endpoint = st.secrets["aws_credentials"]["AWS_ENDPOINT"]

# AWS Session Management
@st.cache_resource 
def get_session(aws_id, aws_secret, aws_token):
    return boto3.Session(
        aws_access_key_id=aws_id,
        aws_secret_access_key=aws_secret,
        aws_session_token=aws_token,
        region_name='us-east-1'
    )

session = get_session(aws_id, aws_secret, aws_token)
sm_session = sagemaker.Session(boto_session=session)

# Data & Model Configuration
MODEL_INFO = {
    "endpoint"  : aws_endpoint,
    "explainer" : "fraud_shap_explainer.pkl", # FIXED: Based on your Jupyter screenshot
    "pipeline"  : "finalized_fraud_model.tar.gz", # FIXED: Based on your Jupyter screenshot
    "keys"      : ['TransactionAmt','addr1','addr2'],
    "inputs"    : [{"name": k, "type": "number", "min": 0.0, "max": 10000.0, "default": 50.0, "step": 1.0} for k in ['TransactionAmt','addr1','addr2']] # FIXED: Realistic bounds
}

def load_pipeline(_session, bucket, key):
    s3_client = _session.client('s3')
    filename = MODEL_INFO["pipeline"]

    s3_client.download_file(
        Filename=filename, 
        Bucket=bucket, 
        Key=f"{key}/{os.path.basename(filename)}"
    )
    
    # Extract the .pkl file from the .tar.gz
    with tarfile.open(filename, "r:gz") as tar:
        tar.extractall(path=".")
        joblib_file = [f for f in tar.getnames() if f.endswith('.pkl')][0]
    
    # Load the full pipeline
    return joblib.load(f"{joblib_file}")

def load_shap_explainer(_session, bucket, key, local_path):
    s3_client = _session.client('s3')

    # Only download if it doesn't exist locally to save time
    if not os.path.exists(local_path):
        s3_client.download_file(Filename=local_path, Bucket=bucket, Key=key)
        
    with open(local_path, "rb") as f:
        return load(f)

# Prediction Logic
def call_model_api(input_dict):
    predictor = Predictor(
        endpoint_name=MODEL_INFO["endpoint"],
        sagemaker_session=sm_session,
        serializer=JSONSerializer(), 
        deserializer=JSONDeserializer() # FIXED: Now matches inference output
    )

    try:
        raw_pred = predictor.predict(input_dict)
        # Handle list wrapping from JSON response safely
        if isinstance(raw_pred, list):
            pred_val = raw_pred[-1]
        else:
            pred_val = raw_pred
            
        mapping = {0: "Legitimate", 1: "Fraud"}
        return mapping.get(pred_val, "Unknown Prediction"), 200
    except Exception as e:
        return f"Error: {str(e)}", 500

# Local Explainability
def display_explanation(input_dict, session, aws_bucket):
    explainer_name = MODEL_INFO["explainer"]
    explainer = load_shap_explainer(session, aws_bucket, posixpath.join('explainer', explainer_name), os.path.join(tempfile.gettempdir(), explainer_name))
    
    best_pipeline = load_pipeline(session, aws_bucket, 'sklearn-pipeline-deployment')
    preprocessing_pipeline = Pipeline(steps=best_pipeline.steps[:-3])
    
    # FIXED: Wrap flat dictionary in a list so pandas doesn't throw a scalar error
    input_df = pd.DataFrame([input_dict]) 
    
    input_df_transformed = preprocessing_pipeline.transform(input_df)
    feature_names = best_pipeline[:-2].get_feature_names_out()
    input_df_transformed = pd.DataFrame(input_df_transformed, columns=feature_names)
    shap_values = explainer(input_df_transformed)
    
    st.subheader("🔍 Decision Transparency (SHAP)")
    fig, ax = plt.subplots(figsize=(10, 4))
    shap.plots.waterfall(shap_values[0, :, 1])  # class 1 = fraud
    st.pyplot(fig)
    top_feature = pd.Series(shap_values[0, :, 1].values, index=shap_values[0, :, 1].feature_names).abs().idxmax()
    st.info(f"**Business Insight:** The most influential factor in this decision was **{top_feature}**.")

# Streamlit UI
st.set_page_config(page_title="ML Deployment", layout="wide")
st.title("👨‍💻 ML Deployment")

with st.form("pred_form"):
    st.subheader(f"Inputs")
    cols = st.columns(2)
    user_inputs = {}
    
    for i, inp in enumerate(MODEL_INFO["inputs"]):
        with cols[i % 2]:
            user_inputs[inp['name']] = st.number_input(
                inp['name'].replace('_', ' ').upper(),
                min_value=inp['min'], max_value=inp['max'], value=inp['default'], step=inp['step']
            )
    
    submitted = st.form_submit_button("Run Prediction")

# FIXED: Extract a flat dictionary from the first row
original = dataset.iloc[0:1].to_dict(orient='records')[0] 
original.update(user_inputs)

if submitted:
    res, status = call_model_api(original)
    if status == 200:
        st.metric("Prediction Result", res)
        display_explanation(original, session, aws_bucket)
    else:
        st.error(res)
