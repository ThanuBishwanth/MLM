import streamlit as st
import pandas as pd
import joblib
import os
import numpy as np
from database.db import get_connection
import io

def load_model_and_data(model_id):
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        SELECT model_file_path, dataset_id, task_type
        FROM models
        WHERE model_id = %s
    """, (model_id,))
    result = cur.fetchone()
    if not result:
        return None, None, None, None

    model_path, dataset_id, task_type = result

    model = joblib.load(model_path)

    cur.execute("SELECT dataset FROM user_datasets WHERE id = %s", (dataset_id,))
    data_csv = cur.fetchone()[0]
    df = pd.read_csv(io.StringIO(data_csv))

    cur.close()
    conn.close()

    return model, df, task_type, dataset_id



def render():
    st.title(" Make Predictions")
    st.markdown(f"{st.session_state["selected_model_name"]} on {st.session_state.dataset}")

    if "selected_model_id" not in st.session_state:
        st.warning("No model selected. Please go to the Models page and choose a model for prediction.")
        st.stop()

    model_id = st.session_state["selected_model_id"]
    model, df, task_type, dataset_id = load_model_and_data(model_id)

    if model is None:
        st.error("Unable to load model or dataset.")
        st.stop()

    st.success(" Model loaded successfully. You can now make predictions.")

    st.markdown("###  Select Target Column")
    target_col = st.selectbox("Which column is the target (the one your model was trained to predict)?", df.columns)

    # Enter feature values
    input_features = [col for col in df.columns if col != target_col]
    st.markdown("###  Enter Input Values for Prediction")

    user_input = {}
    for feature in input_features:
        dtype = df[feature].dtype
        default_val = df[feature].dropna().iloc[0] if not df[feature].dropna().empty else 0
        if np.issubdtype(dtype, np.number):
            user_input[feature] = st.number_input(f"{feature}", value=float(default_val))
        else:
            user_input[feature] = st.text_input(f"{feature}", value=str(default_val))

    if st.button(" Predict"):
        try:
            input_df = pd.DataFrame([user_input])
            prediction = model.predict(input_df)[0]

            
            st.success(f"🎉 Prediction: {prediction}")
        except Exception as e:
            st.error(f"Error during prediction: {e}")
