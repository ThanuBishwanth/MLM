import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report, roc_curve, auc,
    mean_absolute_error, mean_squared_error, r2_score
)
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingRegressor
)
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neural_network import MLPClassifier, MLPRegressor
from pytorch_tabnet.tab_model import TabNetClassifier, TabNetRegressor
from xgboost import XGBClassifier, XGBRegressor
from sklearn.svm import SVR, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import numpy as np
from database.db import get_connection
from skopt import BayesSearchCV
from skopt.space import Real, Integer

def save_model_to_db(user_id, dataset_id, model_choice, task_type, model_path, metrics):
    conn = get_connection()
    cur = conn.cursor()

    metrics = {k: float(v) for k, v in metrics.items()}

    cur.execute("""
        SELECT model_id FROM models
        WHERE user_id = %s AND dataset_id = %s AND model_name = %s AND task_type = %s
    """, (user_id, dataset_id, model_choice, task_type))
    result = cur.fetchone()

    if result:
        model_id = result[0]
        cur.execute("""
            UPDATE models SET model_file_path = %s WHERE model_id = %s
        """, (model_path, model_id))

        if task_type == "Classification":
            cur.execute("""
                UPDATE classification_metrics
                SET accuracy = %s, f1_score = %s, precision = %s, recall = %s
                WHERE model_id = %s
            """, (metrics['accuracy'], metrics['f1'], metrics['precision'], metrics['recall'], model_id))
        else:
            cur.execute("""
                UPDATE regression_metrics
                SET mae = %s, mse = %s, rmse = %s, r2 = %s
                WHERE model_id = %s
            """, (metrics['mae'], metrics['mse'], metrics['rmse'], metrics['r2'], model_id))
    else:
        cur.execute("""
            INSERT INTO models (user_id, dataset_id, model_name, task_type, model_file_path)
            VALUES (%s, %s, %s, %s, %s) RETURNING model_id
        """, (user_id, dataset_id, model_choice, task_type, model_path))
        model_id = cur.fetchone()[0]

        if task_type == "Classification":
            cur.execute("""
                INSERT INTO classification_metrics (model_id, accuracy, f1_score, precision, recall)
                VALUES (%s, %s, %s, %s, %s)
            """, (model_id, metrics['accuracy'], metrics['f1'], metrics['precision'], metrics['recall']))
        else:
            cur.execute("""
                INSERT INTO regression_metrics (model_id, mae, mse, rmse, r2)
                VALUES (%s, %s, %s, %s, %s)
            """, (model_id, metrics['mae'], metrics['mse'], metrics['rmse'], metrics['r2']))

    conn.commit()
    cur.close()
    conn.close()

def recommend_task_type(target_series):
    if target_series.dtype == 'object' or target_series.nunique() < 10:
        return "Classification"
    elif pd.api.types.is_numeric_dtype(target_series) and target_series.nunique() > 15:
        return "Regression"
    else:
        return "Classification"

def recommend_split_ratio(n_samples):
    if n_samples < 200:
        return 0.3
    elif n_samples < 1000:
        return 0.2
    else:
        return 0.1


def render():
    st.title(f"Train Model For {st.session_state.name}")

    if 'user_id' not in st.session_state or 'dataset_id' not in st.session_state:
        st.error("Please login and upload a dataset first.")
        return

    user_id = st.session_state['user_id']
    dataset_id = st.session_state['dataset_id']
    df = st.session_state.get('df')

    if df is None:
        st.warning("No dataset found. Please upload a dataset.")
        return

    target_col = st.selectbox("Select Target Column", df.columns)
    recommended_task = recommend_task_type(df[target_col])
    st.info(f"Recommended Task Type: {recommended_task}")

    task_type = st.radio("Select Task Type", ["Classification", "Regression"],
                         index=0 if recommended_task == "Classification" else 1)

    recommended_split = recommend_split_ratio(len(df))
    st.info(f"Recommended test size: {int(recommended_split * 100)}% for {len(df)} rows.")
    test_size = st.slider("Choose Test Size (%)", min_value=10, max_value=50, value=int(recommended_split * 100), step=5) / 100

    if task_type == "Classification":
        model_options = {
            "Random Forest Classifier": RandomForestClassifier(),
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "Support Vector Classifier (SVC)": SVC(probability=True),
            "K-Nearest Neighbors (KNN)": KNeighborsClassifier(),
            "MLP Classifier (DL)": MLPClassifier(max_iter=1000),
            "TabNet Classifier (DL)": TabNetClassifier(verbose=0),
            "XGBoost Classifier (DL)": XGBClassifier(use_label_encoder=False, eval_metric='logloss')
        }

        search_spaces = {
            "Random Forest Classifier": {
                'n_estimators': Integer(50, 300),
                'max_depth': Integer(3, 20)
            },
            "Logistic Regression": {
                'C': Real(0.01, 10.0, prior='log-uniform')
            },
            "SVC": {
                'C': Real(0.1, 10.0),
                'gamma': Real(0.001, 1.0, prior='log-uniform')
            },
            "XGBoost Classifier (DL)": {
                'n_estimators': Integer(50, 300),
                'learning_rate': Real(0.01, 0.3),
                'max_depth': Integer(3, 10)
            }
        }
    else:
        model_options = {
            "Random Forest Regressor": RandomForestRegressor(),
            "Linear Regression": LinearRegression(),
            "Support Vector Regressor (SVR)": SVR(),
            "Gradient Boosting Regressor": GradientBoostingRegressor(),
            "MLP Regressor (DL)": MLPRegressor(max_iter=1000),
            "XGBoost Regressor (DL)": XGBRegressor()
        }

        search_spaces = {
            "Random Forest Regressor": {
                'n_estimators': Integer(50, 300),
                'max_depth': Integer(3, 20)
            },
            "SVR": {
                'C': Real(0.1, 10.0),
                'gamma': Real(0.001, 1.0, prior='log-uniform')
            },
            "XGBoost Regressor (DL)": {
                'n_estimators': Integer(50, 300),
                'learning_rate': Real(0.01, 0.3),
                'max_depth': Integer(3, 10)
            }
        }

    model_choice = st.selectbox("Select Model", list(model_options.keys()))
    base_model = model_options[model_choice]
    X = df.drop(columns=[target_col])
    y = df[target_col]
    feature_names = X.columns
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    if isinstance(base_model, (LogisticRegression, SVC, KNeighborsClassifier, SVR)):
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    if isinstance(base_model, (TabNetClassifier, TabNetRegressor)):
        X_train = X_train.values
        X_test = X_test.values
        y_train = y_train.values
        y_test = y_test.values

    if st.button("Train Model"):
        model = base_model
        with st.spinner("Training in progress..."):
            if model_choice in search_spaces:
                st.write("Running Bayesian Optimization...")
                opt = BayesSearchCV(
                    model,
                    search_spaces[model_choice],
                    n_iter=25,
                    cv=3,
                    n_jobs=-1,
                    random_state=42
                )
                opt.fit(X_train, y_train)
                st.success("Best hyperparameters found!")
                st.json(opt.best_params_)
                model = opt.best_estimator_
            else:
                model.fit(X_train, y_train)

        st.success(f"{model_choice} training complete!")
        st.subheader("Evaluation Metrics")

        def evaluate_and_display(X_data, y_data, split_label):
            st.markdown(f"### {split_label} Metrics")
            y_pred = model.predict(X_data)
            result = {}

            if task_type == "Classification":
                accuracy = accuracy_score(y_data, y_pred)
                f1 = f1_score(y_data, y_pred, average='weighted')
                precision = precision_score(y_data, y_pred, average='weighted')
                recall = recall_score(y_data, y_pred, average='weighted')
                result = {"accuracy": accuracy, "f1": f1, "precision": precision, "recall": recall}

                cm = confusion_matrix(y_data, y_pred)
                cm_fig = go.Figure(data=go.Heatmap(z=cm, colorscale='Blues'))
                cm_fig.update_layout(title=f"{split_label} Confusion Matrix", autosize=False, width=500, height=500)
                st.plotly_chart(cm_fig, key=f"{split_label}_cm")

                if hasattr(model, "predict_proba"):
                    y_prob = model.predict_proba(X_data)[:, 1]
                    fpr, tpr, _ = roc_curve(y_data, y_prob, pos_label=pd.Series(y_data).unique()[1])
                    roc_auc = auc(fpr, tpr)
                    roc_fig = go.Figure()
                    roc_fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'ROC (AUC={roc_auc:.2f})'))
                    roc_fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', line=dict(dash='dash'), name='Random'))
                    roc_fig.update_layout(title=f'{split_label} ROC Curve', xaxis_title='FPR', yaxis_title='TPR', width=500, height=500)
                    st.plotly_chart(roc_fig, key=f"{split_label}_roc")

                if hasattr(model, "feature_importances_"):
                    importances = model.feature_importances_
                    importance_df = pd.DataFrame({
                        "Feature": feature_names,
                        "Importance": importances
                    }).sort_values(by="Importance", ascending=False)

                    fig = px.bar(importance_df, x="Importance", y="Feature", orientation="h",
                                title="Feature Importance", labels={"Importance": "Score", "Feature": "Features"})
                    fig.update_layout(width=500, height=500)
                    st.plotly_chart(fig, key=f"{split_label}_feature_importance")

                elif hasattr(model, "coef_"):
                    coefs = model.coef_[0] if len(model.coef_.shape) > 1 else model.coef_
                    importance_df = pd.DataFrame({
                        "Feature": feature_names,
                        "Coefficient": coefs
                    }).sort_values(by="Coefficient", key=abs, ascending=False)

                    fig = px.bar(importance_df, x="Coefficient", y="Feature", orientation="h",
                                title="Model Coefficients", labels={"Coefficient": "Value", "Feature": "Features"})
                    fig.update_layout(width=500, height=500)
                    st.plotly_chart(fig, key=f"{split_label}_coefficients")

                
            else:
                mae = mean_absolute_error(y_data, y_pred)
                mse = mean_squared_error(y_data, y_pred)
                rmse = np.sqrt(mse)
                r2 = r2_score(y_data, y_pred)
                result = {"mae": mae, "mse": mse, "rmse": rmse, "r2": r2}

                scatter_fig = px.scatter(x=y_data, y=y_pred, labels={"x": "Actual", "y": "Predicted"},
                                        title=f"{split_label} Actual vs Predicted")
                scatter_fig.update_layout(width=500, height=500)
                st.plotly_chart(scatter_fig, key=f"{split_label}_scatter")

                residuals = y_data - y_pred
                residual_fig = px.scatter(x=y_pred, y=residuals, labels={"x": "Predicted", "y": "Residuals"},
                                        title=f"{split_label} Residual Plot")
                residual_fig.update_layout(width=500, height=500)
                st.plotly_chart(residual_fig, key=f"{split_label}_residuals")

                if hasattr(model, "feature_importances_"):
                    importances = model.feature_importances_
                    importance_df = pd.DataFrame({
                        "Feature": feature_names,
                        "Importance": importances
                    }).sort_values(by="Importance", ascending=False)

                    fig = px.bar(importance_df, x="Importance", y="Feature", orientation="h",
                                title="Feature Importance", labels={"Importance": "Score", "Feature": "Features"})
                    fig.update_layout(width=500, height=500)
                    st.plotly_chart(fig, key=f"{split_label}_feature_importance")

                elif hasattr(model, "coef_"):
                    coefs = model.coef_[0] if len(model.coef_.shape) > 1 else model.coef_
                    importance_df = pd.DataFrame({
                        "Feature": feature_names,
                        "Coefficient": coefs
                    }).sort_values(by="Coefficient", key=abs, ascending=False)

                    fig = px.bar(importance_df, x="Coefficient", y="Feature", orientation="h",
                                title="Model Coefficients", labels={"Coefficient": "Value", "Feature": "Features"})
                    fig.update_layout(width=500, height=500)
                    st.plotly_chart(fig, key=f"{split_label}_coefficients")

                

            return result

        tabs = st.tabs(["Train Metrics", "Test Metrics"])

        with tabs[0]:
            st.markdown("### 📊 Training Evaluation")
            train_metrics = evaluate_and_display(X_train, y_train, "Train")
            st.table(pd.DataFrame(train_metrics, index=["Score"]).T)

        with tabs[1]:
            st.markdown("### 📊 Test Evaluation")
            test_metrics = evaluate_and_display(X_test, y_test, "Test")
            st.table(pd.DataFrame(test_metrics, index=["Score"]).T)

        model_filename = f"{model_choice.replace(' ', '_')}_{user_id}_{dataset_id}.joblib"
        joblib.dump(model, model_filename)
        save_model_to_db(user_id, dataset_id, model_choice, task_type, model_filename, test_metrics)

        with open(model_filename, "rb") as f:
            st.download_button("Download Trained Model", f, model_filename, mime="application/octet-stream")
