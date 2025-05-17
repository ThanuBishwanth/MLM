import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, SpectralClustering, MeanShift
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from database.db import get_connection
import os
os.environ["OMP_NUM_THREADS"] = "5"
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score


# ------------------ Decision Metrics for Comparison
def calculate_comparison_metrics(X, cluster_labels):
    metrics = {}
    
    # Calculate Silhouette Score
    metrics['Silhouette Score'] = silhouette_score(X, cluster_labels)
    
    # Calculate Calinski-Harabasz Index
    metrics['Calinski-Harabasz Index'] = calinski_harabasz_score(X, cluster_labels)
    
    # Calculate Davies-Bouldin Index
    metrics['Davies-Bouldin Index'] = davies_bouldin_score(X, cluster_labels)
    
    return metrics
# ------------------ Preprocessing
def preprocess_data(df):
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df)
    return df_scaled

# ------------------ Elbow Plot
def plot_elbow(X):
    sse = []
    k_range = range(2, 11)
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X)
        sse.append(kmeans.inertia_)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(k_range, sse, marker='o')
    ax.set_xlabel('Number of clusters')
    ax.set_ylabel('SSE (Inertia)')
    ax.set_title('Elbow Method For Optimal k')
    st.pyplot(fig)

# ------------------ PCA Visualization
def visualize_pca(X, labels, title):
    pca = PCA(n_components=2)
    components = pca.fit_transform(X)
    df_pca = pd.DataFrame(components, columns=['PC1', 'PC2'])
    df_pca['Cluster'] = labels

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(data=df_pca, x='PC1', y='PC2', hue='Cluster', palette='tab10', s=80, ax=ax)
    ax.set_title(title)
    ax.legend()
    st.pyplot(fig)

# ------------------ t-SNE Visualization
def visualize_tsne(X, labels, title):
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    components = tsne.fit_transform(X)
    df_tsne = pd.DataFrame(components, columns=['Dim1', 'Dim2'])
    df_tsne['Cluster'] = labels

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(data=df_tsne, x='Dim1', y='Dim2', hue='Cluster', palette='tab10', s=80, ax=ax)
    ax.set_title(title)
    ax.legend()
    st.pyplot(fig)


# ------------------ Silhouette Score
def plot_silhouette(X, cluster_labels):
    score = silhouette_score(X, cluster_labels)
    st.write(f"Silhouette Score: **{score:.4f}**")



# ------------------ DBSCAN Outlier Detection
def detect_outliers_dbscan(X, eps, min_samples):
    db = DBSCAN(eps=eps, min_samples=min_samples)
    labels = db.fit_predict(X)
    outliers = np.sum(labels == -1)
    return outliers



# ------------------ Load original dataset
def load_dataset_from_db(user_id, dataset_id):
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute("SELECT dataset FROM user_datasets WHERE user_id = %s and id = %s", (user_id, dataset_id))
        result = cur.fetchone()
        cur.close()
        conn.close()
        if result:
            df = pd.read_csv(io.StringIO(result[0]))
            return df
    except Exception as e:
        st.error("Error loading dataset from DB: " + str(e))
    return None

# ------------------ Main Render
def render():
    st.title("Clustering Analysis")

    if not st.session_state.get("authenticated"):
        st.error("Please login to access this page.")
        return

    user_id = st.session_state['user_id']

    # Load dataset
    if st.session_state.df is None:
        df = load_dataset_from_db(user_id, st.session_state["dataset_id"])
        if df is None:
            st.warning("No dataset found. Please upload a dataset first.")
            return
        st.session_state.df = df
    df = st.session_state.df

    num_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()

    if not num_cols:
        st.error("No numeric columns available for clustering.")
        return

    # ----------- Step 1: Feature Distribution
    with st.expander(" Feature Distribution", expanded=False):
        selected_feature = st.selectbox("Select a feature to visualize", num_cols)
        if st.button("Show"):
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.histplot(df[selected_feature], kde=True, ax=ax)
            st.pyplot(fig)

    # ----------- Step 2: Preprocessing
    with st.expander("Preprocessing", expanded=False):
        if st.button("Standardize Data"):
            X = preprocess_data(df[num_cols])
            st.success("Standardized")
        else:
            X = df[num_cols].values
        st.session_state["cluster_X"] = X

    # ----------- Step 3: Algorithm Selection
    with st.expander("Choose Clustering Algorithm", expanded=st.session_state.get("algo_expanded", False)):
        clustering_algorithm = st.selectbox("Algorithm", ["K-Means", "DBSCAN", "Agglomerative Clustering", "Spectral Clustering", "Mean Shift"])
        
        if st.button("Choose"):
            st.session_state["clustering_algorithm"] = clustering_algorithm
            st.session_state["algo_expanded"] = True  # Keep the expander open
            st.session_state["cluster_params"] = {}   # Reset params

        # Check if algorithm is selected and in session state
        if "clustering_algorithm" in st.session_state:
            selected_algo = st.session_state["clustering_algorithm"]
            params = st.session_state.get("cluster_params", {})

            if selected_algo in ["K-Means", "Agglomerative Clustering", "Spectral Clustering"]:
                plot_elbow(st.session_state["cluster_X"])
                p = st.text_input("Enter number of clusters (2-10)", key="n_clusters_input")
                if st.button("Set", key="set_n_clusters"):
                    if p.isdigit() and 2 <= int(p) <= 10:
                        params['n_clusters'] = int(p)
                        st.session_state["cluster_params"] = params
                        st.success(f"Number of clusters set to {p}")
                    else:
                        st.error("Please enter a valid number between 2 and 10")

                
                


            if clustering_algorithm == "DBSCAN":
                params['eps'] = st.slider("Epsilon (eps)", 0.1, 5.0, 0.5)
                params['min_samples'] = st.slider("Min Samples", 2, 10, 5)
                if st.checkbox("Detect Outliers (DBSCAN only)"):
                    outliers = detect_outliers_dbscan(st.session_state["cluster_X"], params['eps'], params['min_samples'])
                    st.info(f"Outliers detected: {outliers}")

            st.session_state["cluster_params"] = params

    # ----------- Step 4: Execute Clustering
    with st.expander("Run Clustering", expanded=False):
        if st.button("Execute Clustering"):
            X = st.session_state["cluster_X"]
            algo = st.session_state["clustering_algorithm"]
            params = st.session_state["cluster_params"]

            if algo == "K-Means":
                model = KMeans(n_clusters=params['n_clusters'], random_state=42)
            elif algo == "DBSCAN":
                model = DBSCAN(eps=params['eps'], min_samples=params['min_samples'])
            elif algo == "Agglomerative Clustering":
                model = AgglomerativeClustering(n_clusters=params['n_clusters'])
            elif algo == "Spectral Clustering":
                model = SpectralClustering(n_clusters=params['n_clusters'], assign_labels="discretize", random_state=42)
            elif algo == "Mean Shift":
                model = MeanShift()

            cluster_labels = model.fit_predict(X)
            df["Cluster"] = cluster_labels
            st.session_state.df = df
            st.session_state["cluster_labels"] = cluster_labels

            st.success("✅ Clustering done!")
            metrics = calculate_comparison_metrics(X, cluster_labels)

            metrics_df = pd.DataFrame([metrics])

            st.dataframe(metrics_df)


            # Show cluster sizes
            st.subheader("📊 Cluster Sizes")
            st.dataframe(df['Cluster'].value_counts().rename_axis('Cluster').reset_index(name='Counts'))


    # ----------- Step 5: Visualizations
    with st.expander(" Visualizations", expanded=False):
        if "cluster_labels" not in st.session_state:
            st.info("Please run clustering first.")
        else:
            X = st.session_state["cluster_X"]
            cluster_labels = st.session_state["cluster_labels"]
            algo = st.session_state["clustering_algorithm"]

            st.subheader(" PCA")
            visualize_pca(X, cluster_labels, f"PCA - {algo}")

            st.subheader(" t-SNE")
            visualize_tsne(X, cluster_labels, f"t-SNE - {algo}")

            st.subheader(" Cluster Interpretation & Notes")
            cluster_names = {}
            unique_clusters = sorted(df["Cluster"].unique())
            for cluster in unique_clusters:
                name = st.text_input(f"Name for Cluster {cluster}", value=f"Cluster {cluster}")
                cluster_names[str(cluster)] = name

            notes = st.text_area("General Notes / Decisions", placeholder="Summarize key patterns, outliers, cluster roles, next steps...")

    # ----------- Step 6: Cluster Profiling
    with st.expander(" Cluster Profiling", expanded=False):
        if "Cluster" in df.columns:
            if st.button("Show Cluster Profiles"):
                profile = df.groupby("Cluster").mean(numeric_only=True).reset_index()
                numeric_cols = profile.select_dtypes(include=["number"]).columns
                if len(numeric_cols) > 0:
                    st.dataframe(profile.style.highlight_max(axis=1, subset=numeric_cols))
                else:
                    st.dataframe(profile)


    # ----------- Step 7: Rename Clusters
    with st.expander(" Rename Clusters", expanded=False):
        if "Cluster" in df.columns:
            unique_clusters = sorted(df["Cluster"].unique())
            rename_mapping = {}
            for c in unique_clusters:
                new_name = st.text_input(f"Rename Cluster {c} to", value=str(c))
                rename_mapping[c] = new_name
            if st.button("Apply Renaming"):
                df["Cluster"] = df["Cluster"].map(rename_mapping)
                st.session_state.df = df
                st.success("Cluster names updated!")
                st.dataframe(df.head())
    
     # ----------- Step 6: Decision Making
        # ----------- Step 9: Decision Making & Algorithm Comparison
    with st.expander(" Decision Making & Algorithm Comparison", expanded=True):
        st.markdown("This section helps you **compare clustering algorithms** based on multiple metrics and choose the most suitable one.")

        if st.button("Compare Algorithms"):
            from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score
            X = st.session_state["cluster_X"]

            algorithms = {
                "KMeans": KMeans(n_clusters=st.session_state["cluster_params"].get('n_clusters', 3), random_state=42),
                "Agglomerative": AgglomerativeClustering(n_clusters=st.session_state["cluster_params"].get('n_clusters', 3)),
                "Spectral": SpectralClustering(n_clusters=st.session_state["cluster_params"].get('n_clusters', 3), assign_labels="discretize", random_state=42),
                "DBSCAN": DBSCAN(eps=0.5, min_samples=5),
                "MeanShift": MeanShift()
            }

            metrics_list = []
            for name, algo in algorithms.items():
                try:
                    labels = algo.fit_predict(X)
                    if len(set(labels)) < 2 or len(set(labels)) == len(X):
                        continue
                    sil = silhouette_score(X, labels)
                    ch = calinski_harabasz_score(X, labels)
                    db = davies_bouldin_score(X, labels)
                    metrics_list.append((name, sil, ch, db))
                except Exception as e:
                    continue

            if not metrics_list:
                st.error("Comparison failed. Try adjusting parameters (like cluster count or DBSCAN settings).")
            else:
                metrics_df = pd.DataFrame(metrics_list, columns=["Algorithm", "Silhouette Score", "Calinski-Harabasz", "Davies-Bouldin"])
                st.subheader("📊 Comparison Table")
                st.dataframe(metrics_df.style.format(precision=4).highlight_max(axis=0, subset=["Silhouette Score", "Calinski-Harabasz"])
                                                      .highlight_min(axis=0, subset=["Davies-Bouldin"]))

                # Visualization
                st.subheader("📈 Comparison Charts")
                fig, ax = plt.subplots(1, 3, figsize=(18, 5))
                labels = metrics_df["Algorithm"]
                ax[0].bar(labels, metrics_df["Silhouette Score"], color='skyblue')
                ax[0].set_title("Silhouette Score ↑")
                ax[1].bar(labels, metrics_df["Calinski-Harabasz"], color='lightgreen')
                ax[1].set_title("Calinski-Harabasz ↑")
                ax[2].bar(labels, metrics_df["Davies-Bouldin"], color='salmon')
                ax[2].set_title("Davies-Bouldin ↓")
                for a in ax:
                    a.tick_params(axis='x', rotation=15)
                st.pyplot(fig)

                # Recommendation
                best_algo = metrics_df.sort_values(by=["Silhouette Score", "Calinski-Harabasz", "Davies-Bouldin"], ascending=[False, False, True]).iloc[0]["Algorithm"]
                st.success(f"✅ **Recommended Algorithm:** {best_algo} based on best combination of metrics.")

        st.caption("Use this to guide your decision before finalizing and saving your clustering results.")

