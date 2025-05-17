import streamlit as st
import pandas as pd
import os
from io import StringIO, BytesIO
from database.db import get_connection
import json
import arff  
import sqlite3
from xml.etree import ElementTree as ET

SUPPORTED_FORMATS = ["csv", "tsv", "txt", "dat", "json", "xml", "arff", "xlsx", "xls", "ods", "db"]


def load_file(file, filetype):
    try:
        if filetype in ["csv"]:
            return pd.read_csv(file)
        elif filetype in ["tsv", "txt", "dat"]:
            return pd.read_csv(file, sep=None, engine='python')
        elif filetype in ["xlsx", "xls"]:
            return pd.read_excel(file, engine="openpyxl")
        elif filetype == "ods":
            return pd.read_excel(file, engine="odf")
        elif filetype == "json":
            return pd.json_normalize(json.load(file))
        elif filetype == "xml":
            tree = ET.parse(file)
            root = tree.getroot()
            data = [child.attrib for child in root]
            return pd.DataFrame(data)
        elif filetype == "arff":
            decoded = file.read().decode("utf-8")
            arff_data = arff.loads(decoded)
            return pd.DataFrame(arff_data["data"], columns=[a[0] for a in arff_data["attributes"]])
        elif filetype == "db":
            temp_path = "/tmp/temp.db"
            with open(temp_path, "wb") as f:
                f.write(file.read())
            conn = sqlite3.connect(temp_path)
            tables = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table';", conn)
            table_name = tables.iloc[0, 0] if not tables.empty else None
            if not table_name:
                raise Exception("No tables found in database.")
            return pd.read_sql(f"SELECT * FROM {table_name}", conn)
        else:
            raise ValueError("Unsupported format.")
    except Exception as e:
        raise Exception(f"Error parsing file ({filetype}): {e}")

def render():
    if not st.session_state.get("authenticated"):
        st.error("Please log in to access this page.")
        return

    st.title("Upload Dataset")

    user_id = st.session_state.get("user_id")
    file = st.file_uploader("Choose a dataset file", type=SUPPORTED_FORMATS)
    name = st.text_input("Dataset name (Must be unique)")
    st.session_state.name=name

    if st.button("Upload"):
        if not file:
            st.warning("Please select a dataset file.")
            return

        if not name.strip():
            st.warning("Please provide a dataset name.")
            return

        try:
            filetype = file.name.split(".")[-1].lower()
            df = load_file(file, filetype)
            csv_data = df.to_csv(index=False)

            conn = get_connection()
            cur = conn.cursor()

            # Check if the dataset name already exists for the current user
            cur.execute("""
                SELECT COUNT(*) FROM user_datasets
                WHERE user_id = %s AND dataset_name = %s
            """, (user_id, name.strip()))
            if cur.fetchone()[0] > 0:
                st.error("Dataset name already exists. Please choose a different name.")
                cur.close()
                conn.close()
                return

            # Insert the new dataset into the database
            cur.execute("""
                INSERT INTO user_datasets (user_id, dataset_name, dataset, uploaded_at)
                VALUES (%s, %s, %s, NOW())
            """, (user_id, name.strip(), csv_data))
            conn.commit()

            # Fetch the ID of the newly uploaded dataset
            cur.execute("SELECT id FROM user_datasets WHERE user_id=%s AND dataset_name=%s",
                        (user_id, name.strip()))
            st.session_state["dataset_id"] = cur.fetchone()[0]
            st.session_state["df"] = None

            cur.close()
            conn.close()

            st.success("✅ Dataset uploaded successfully.")
            st.dataframe(df.head())

        except Exception as e:
            st.error(f"Upload failed: {e}")

    st.markdown("---")
    st.subheader("Previously Uploaded Datasets")

    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute("""
            SELECT id, dataset_name, uploaded_at
            FROM user_datasets
            WHERE user_id = %s
            ORDER BY uploaded_at DESC
        """, (user_id,))
        datasets = cur.fetchall()
        cur.close()
        conn.close()

        if datasets:
            for dataset_id, dataset_name, uploaded_at in datasets:
                df = None
                col1, col2, col3 = st.columns([6, 1, 1])

                with col1:
                    st.markdown(
                        f"**{dataset_name}**  &nbsp;&nbsp; _(uploaded {uploaded_at.strftime('%Y-%m-%d %H:%M')})_",
                        unsafe_allow_html=True
                    )

                with col2:
                    if st.button("♻️ Reuse", key=f"reuse_{dataset_id}"):
                        conn = get_connection()
                        cur = conn.cursor()
                        cur.execute("SELECT dataset FROM user_datasets WHERE id = %s", (dataset_id,))
                        dataset_csv = cur.fetchone()[0]
                        cur.execute("SELECT dataset_name FROM user_datasets WHERE id = %s", (dataset_id,))
                        st.session_state.name=cur.fetchone()[0]
                        cur.close()
                        conn.close()
                        df = pd.read_csv(StringIO(dataset_csv))
                        st.session_state["dataset_id"] = dataset_id
                        st.session_state["df"] = df
                        st.toast(f"✅ Loaded dataset: {dataset_name}")

                with col3:
                    if st.button("🗑️ Delete", key=f"delete_{dataset_id}"):
                        conn = get_connection()
                        cur = conn.cursor()
                        cur.execute("DELETE FROM user_datasets WHERE id = %s AND user_id = %s", (dataset_id, user_id))
                        conn.commit()
                        cur.close()
                        conn.close()
                        st.toast(f"🗑️ Deleted dataset: {dataset_name}")
                        st.rerun()

                if df is not None:
                    st.dataframe(df.head())

                st.markdown("<hr style='border:1px solid #bbb;margin: 5px 0px;'>", unsafe_allow_html=True)
        else:
            st.info("No datasets uploaded yet.")

    except Exception as e:
        st.error(f"Error fetching uploaded datasets: {e}")
