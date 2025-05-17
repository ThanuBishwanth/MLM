import streamlit as st
import pandas as pd
import psycopg2
import os
import joblib
from datetime import datetime
from database.db import get_connection, get_sqlalchemy_engine
import plotly.express as px

# Fetch classification metrics for a model
def fetch_classification_metrics(conn, model_id):
    with conn.cursor() as cur:
        cur.execute("""
            SELECT accuracy, f1_score, precision, recall
            FROM classification_metrics
            WHERE model_id = %s
        """, (model_id,))
        return cur.fetchone()

# Fetch regression metrics for a model
def fetch_regression_metrics(conn, model_id):
    with conn.cursor() as cur:
        cur.execute("""
            SELECT mae, mse, rmse, r2
            FROM regression_metrics
            WHERE model_id = %s
        """, (model_id,))
        return cur.fetchone()

# Delete model and its associated metrics
def delete_model_and_metrics(conn, model_id, task_type):
    with conn.cursor() as cur:
        if task_type == "Classification":
            cur.execute("DELETE FROM classification_metrics WHERE model_id = %s", (model_id,))
        else:
            cur.execute("DELETE FROM regression_metrics WHERE model_id = %s", (model_id,))
        cur.execute("DELETE FROM models WHERE model_id = %s", (model_id,))
        conn.commit()

# Get the best classification model based on average score
def get_best_classification_model(models):
    for m in models:
        m['avg_score'] = (m['accuracy'] + m['f1_score'] + m['precision'] + m['recall']) / 4
    best = max(models, key=lambda m: m['avg_score'])
    reason = (
        f"This model has the highest average score across all metrics "
        f"(Accuracy: {best['accuracy']:.2f}, F1: {best['f1_score']:.2f}, "
        f"Precision: {best['precision']:.2f}, Recall: {best['recall']:.2f}) "
        f"with an average of {best['avg_score']:.2f}."
    )
    return best, reason

# Get the best regression model based on inverse average error
def get_best_regression_model(models):
    for m in models:
        m['inv_avg_error'] = (1 / (m['mae'] + 1e-6) + 1 / (m['mse'] + 1e-6) +
                              1 / (m['rmse'] + 1e-6) + m['r2']) / 4
    best = max(models, key=lambda m: m['inv_avg_error'])
    reason = (
        f"This model has the best combined average of low error metrics "
        f"(MAE: {best['mae']:.2f}, MSE: {best['mse']:.2f}, RMSE: {best['rmse']:.2f}, R²: {best['r2']:.2f}) "
        f"resulting in the highest score ({best['inv_avg_error']:.2f}) when considering all."
    )
    return best, reason

# Fetch all model metrics for comparison chart
def fetch_all_models_metrics(user_id, dataset_id, task_type):
    conn = get_connection()
    cur = conn.cursor()

    if task_type == "Classification":
        cur.execute("""
            SELECT m.model_name, c.accuracy, c.f1_score, c.precision, c.recall
            FROM models m
            JOIN classification_metrics c ON m.model_id = c.model_id
            WHERE m.user_id = %s AND m.dataset_id = %s AND m.task_type = %s
        """, (user_id, dataset_id, task_type))
        columns = ["Model", "Accuracy", "F1 Score", "Precision", "Recall"]
    else:
        cur.execute("""
            SELECT m.model_name, r.mae, r.mse, r.rmse, r.r2
            FROM models m
            JOIN regression_metrics r ON m.model_id = r.model_id
            WHERE m.user_id = %s AND m.dataset_id = %s AND m.task_type = %s
        """, (user_id, dataset_id, task_type))
        columns = ["Model", "MAE", "MSE", "RMSE", "R²"]

    rows = cur.fetchall()
    cur.close()
    conn.close()
    return pd.DataFrame(rows, columns=columns)

# Main render function
def render():
    st.title("Your Trained Models")

    if "user_id" not in st.session_state:
        st.error("You must be logged in to view this page.")
        return

    user_id = st.session_state["user_id"]
    conn = get_connection()

    df_datasets = pd.read_sql("""
        SELECT id AS dataset_id, dataset_name
        FROM user_datasets
        WHERE user_id = %s
    """, conn, params=(user_id,))

    if df_datasets.empty:
        st.info("You haven’t trained any models yet. Go to the Train page to get started!")
        return

    for _, row in df_datasets.iterrows():
        dataset_id = row["dataset_id"]
        dataset_name = row["dataset_name"]
        st.markdown(f"## Dataset: {dataset_name}")

        df_models = pd.read_sql("""
            SELECT model_id, model_name, task_type, model_file_path, created_at
            FROM models
            WHERE user_id = %s AND dataset_id = %s
            ORDER BY created_at DESC
        """, conn, params=(user_id, dataset_id))

        if df_models.empty:
            st.info("No models trained on this dataset yet.")
            continue

        for task_type in ["Classification", "Regression"]:
            filtered_models = df_models[df_models['task_type'] == task_type]
            if filtered_models.empty:
                continue

            with st.expander(f"**{task_type} Models ({len(filtered_models)})**", expanded=False):
                model_metrics = []
                count = 0

                for _, model_row in filtered_models.iterrows():
                    count += 1
                    model_id = model_row["model_id"]
                    model_name = model_row["model_name"]
                    model_path = model_row["model_file_path"]
                    created_at = model_row["created_at"].strftime('%Y-%m-%d %H:%M')
                    st.markdown("---")
                    st.markdown(f"**{count}. {model_name} (Trained on {created_at})**")

                    col1, col2, col3 = st.columns(3)

                    with col1:
                        try:
                            with open(model_path, "rb") as f:
                                st.download_button("Download", data=f, file_name=os.path.basename(model_path))
                        except FileNotFoundError:
                            st.warning("Model file not found.")

                    with col2:
                        if st.button("Use for Prediction", key=f"use_{model_id}"):
                            try:
                                model = joblib.load(model_path)
                                st.session_state["selected_model_id"] = model_id
                                st.session_state["selected_model"] = model
                                st.session_state["selected_model_name"] = model_name
                                st.session_state["selected_model_task"] = task_type
                                st.session_state.dataset=dataset_name
                                st.toast("Model loaded! You can now go to the Prediction page.")
                                st.success("Model loaded. Go to the Prediction page to start predicting.")
                            except Exception as e:
                                st.error("Failed to load model file.")

                    with col3:
                        if st.button("Delete", key=f"delete_{model_id}"):
                            delete_model_and_metrics(conn, model_id, task_type)
                            if os.path.exists(model_path):
                                os.remove(model_path)
                            st.toast("Model deleted successfully.")
                            st.rerun()

                    if task_type == "Classification":
                        metrics = fetch_classification_metrics(conn, model_id)
                        if metrics:
                            accuracy, f1, precision, recall = metrics
                            st.markdown(f"Accuracy: {accuracy:.2f} | F1 Score: {f1:.2f} | Precision: {precision:.2f} | Recall: {recall:.2f}")
                            model_metrics.append({
                                "model_id": model_id,
                                "model_name": model_name,
                                "f1_score": f1,
                                "accuracy": accuracy,
                                "precision": precision,
                                "recall": recall
                            })
                    else:
                        metrics = fetch_regression_metrics(conn, model_id)
                        if metrics:
                            mae, mse, rmse, r2 = metrics
                            st.markdown(f"R²: {r2:.2f} | RMSE: {rmse:.2f} | MAE: {mae:.2f} | MSE: {mse:.2f}")
                            model_metrics.append({
                                "model_id": model_id,
                                "model_name": model_name,
                                "rmse": rmse,
                                "mae": mae,
                                "mse": mse,
                                "r2": r2
                            })

                if model_metrics:
                    st.divider()
                    if task_type == "Classification":
                        best, reason = get_best_classification_model(model_metrics)
                    else:
                        best, reason = get_best_regression_model(model_metrics)

                    st.markdown(f"**Recommended {task_type} Model:** {best['model_name']}")
                    st.caption(reason)

                st.markdown("**Model Comparison**")
                df_metrics = fetch_all_models_metrics(user_id, dataset_id, task_type)

                if not df_metrics.empty:
                    melted = df_metrics.melt(id_vars="Model", var_name="Metric", value_name="Score")
                    fig = px.bar(melted, x="Model", y="Score", color="Metric", barmode="group", title=f"{task_type} Model Metrics Comparison")

                    # 🛠 FIX: Add a unique key for the chart
                    st.plotly_chart(fig, use_container_width=True, key=f"{task_type}_chart_{dataset_id}")
                    st.caption("Interpretation: This chart helps you visually compare the performance of all trained models on the selected dataset.")

    conn.close()
