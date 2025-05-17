import streamlit as st
import pandas as pd
import psycopg2
import io
from datetime import datetime
from database.db import get_connection
import plotly.express as px
import numpy as np

def insert_encoding_map(dataset_id, column_name, original_value, encoded_value):
    conn = get_connection()
    cur = conn.cursor()
    query = """
    INSERT INTO encoding_maps (dataset_id, column_name, original_value, encoded_value)
    VALUES (%s, %s, %s, %s)
    ON CONFLICT DO NOTHING;  -- Optional, if duplicates possible
    """
    cur.execute(query, (dataset_id, column_name, original_value, encoded_value))
    conn.commit()
    cur.close()
    conn.close()


def update_dataset_in_db(user_id, df,dataset_id):
    try:
        conn = get_connection()
        cur = conn.cursor()
        df_csv_str = df.to_csv(index=False)
        cur.execute("UPDATE user_datasets SET dataset = %s, uploaded_at = %s WHERE user_id = %s and id=%s",
                    (df_csv_str, datetime.utcnow(), user_id,dataset_id))
        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        st.error("Error updating dataset in DB: " + str(e))

def load_dataset_from_db(user_id,dataset_id):
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute("SELECT dataset FROM user_datasets WHERE user_id = %s and id=%s", (user_id,dataset_id))
        result = cur.fetchone()
        cur.close()
        conn.close()
        if result:
            df = pd.read_csv(io.StringIO(result[0]))
            return df
    except Exception as e:
        st.error("Error loading dataset from DB: " + str(e))
    return None

def render():
    
    

    if not st.session_state.get("authenticated"):
        st.error("Please login to access this page.")
        return

    user_id = st.session_state['user_id']
    # Load dataset into session state if not already
    if st.session_state.df is None:
        df = load_dataset_from_db(user_id,st.session_state["dataset_id"])
        if df is None:
            st.warning("No dataset found. Please upload a dataset first.")
            return
        st.session_state.df=df
    df=st.session_state.df
    st.title(f"Preprocess Dataset: {st.session_state.name}")

    # Step 1: Load CSV
    with st.expander("Load CSV"):
        st.code('df = pd.read_csv("your_dataset.csv")', language="python")
        if st.button("Execute Load CSV"):
            st.success("Dataset loaded successfully.")
            st.dataframe(df.head())
        # First, clean the invalid values:
        df.replace(["inf", "-inf", "Inf", "-Inf", "#NAME?", "#Name?", "#N/A", "N/A", "nan", "NaN"], np.nan, inplace=True)

        # Also, convert any infinite numbers to NaN
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    

    #Step 2: Data Info
    with st.expander("Dataset Information"):
        st.code("df.info()",language="python")
        if st.button("Execute Command"):
            buffer=io.StringIO()
            df.info(buf=buffer)
            st.text(buffer.getvalue())

    # Step 3: Describe Dataset
    with st.expander("Describe Dataset"):
        st.code("df.describe()", language="python")
        if st.button(" Execute Describe"):
            st.dataframe(df.describe())
    with st.expander("Auto-Detect Date/Time Columns and Handle"):
        date_columns = df.select_dtypes(include=['datetime', 'datetime64']).columns.tolist()

        if date_columns:
            st.write(f"### 📅 Detected Date/Time Columns: {', '.join(date_columns)}")
            for date_col in date_columns:
                st.markdown(f"**Column `{date_col}`** is a Date/Time column.")
                st.write(f"**Example values:**")
                st.write(df[date_col].head())

                # Suggest converting to Date/Time format if needed
                if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
                    st.write(f"**🔧 Suggested Action:** Converting `{date_col}` to Date/Time format.")
                    if st.button(f"Convert `{date_col}` to Date/Time", key=f"convert_{date_col}"):
                        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
                        st.success(f"✅ `{date_col}` converted to Date/Time format.")
                        st.dataframe(df[date_col].head())
                        st.session_state.df = df
                        update_dataset_in_db(user_id, df, st.session_state["dataset_id"])

        else:
            st.success("🎯 No Date/Time columns detected!")
        # Step 7: Drop Unwanted Columns
    with st.expander("Remove Unwanted Columns"):
        cols_to_remove = st.multiselect("Select columns to drop", options=df.columns.tolist())
        if st.button("Execute Drop Columns"):
            if cols_to_remove:
                df = df.drop(columns=cols_to_remove)
                st.success(f"Dropped columns: {', '.join(cols_to_remove)}")
                st.dataframe(df.head())
                st.session_state.df = df
                update_dataset_in_db(user_id, df,st.session_state["dataset_id"])
            else:
                st.info("No columns selected.")
    
    with st.expander("Handle Non-Numeric Text in Numeric Columns (Smart Detection + Recommendation)"):
        # Detect suspicious numeric-looking text columns
        mixed_cols = []

        for col in df.columns:
            if df[col].dtype == 'object':
                # 🌟 Early cleaning for known invalids and infinities
                invalid_entries = ['#NAME?', '#VALUE!', '#DIV/0!', 'N/A', 'NA', '#N/A', 'null', 'NULL', 'inf', '-inf', 'Inf', '-Inf']
                df[col] = df[col].replace(invalid_entries, pd.NA)

                # 🌟 Try auto-convert to numeric if possible
                try_numeric = pd.to_numeric(df[col], errors='coerce')
                numeric_ratio = try_numeric.notnull().mean()

                if 0.9 < numeric_ratio:  # >90% numeric → auto-clean
                    df[col] = try_numeric
                    continue

                # Otherwise, mixed detection
                num_like = df[col].astype(str).str.extract(r'([-+]?\d*\.\d+|\d+)')
                num_like_count = num_like.notnull().sum()[0]
                total_count = len(df[col])
                if 0 < num_like_count < total_count:
                    mixed_cols.append((col, num_like_count / total_count))  # save percentage

        if mixed_cols:
            col_names = [col for col, _ in mixed_cols]
            selected_col = st.selectbox("Select a column with mixed numeric/text data", col_names)

            # Get numeric ratio for selected column
            selected_col_ratio = dict(mixed_cols)[selected_col]

            # Auto Suggest
            if selected_col_ratio > 0.95:
                suggestion = "Set Non-Numeric as NaN"
            elif selected_col_ratio < 0.7:
                suggestion = "Drop Rows with Text"
            else:
                suggestion = "Extract Numbers"

            st.info(f"💡 Recommended action for `{selected_col}`: **{suggestion}**")

            # Let user pick manually still
            action = st.radio("Choose how to handle:", 
                            ["Extract Numbers", "Set Non-Numeric as NaN", "Drop Rows with Text"], 
                            index=["Extract Numbers", "Set Non-Numeric as NaN", "Drop Rows with Text"].index(suggestion))

            if st.button("Apply Handling on Mixed Text Column"):
                if action == "Extract Numbers":
                    df[selected_col] = df[selected_col].astype(str).str.extract(r'([-+]?\d*\.\d+|\d+)')
                    df[selected_col] = pd.to_numeric(df[selected_col], errors='coerce')
                    st.success(f"Extracted numeric parts from `{selected_col}`.")

                elif action == "Set Non-Numeric as NaN":
                    df[selected_col] = pd.to_numeric(df[selected_col], errors='coerce')
                    st.success(f"Set non-numeric values as NaN in `{selected_col}`.")

                elif action == "Drop Rows with Text":
                    df = df[df[selected_col].astype(str).str.match(r'^[-+]?\d*\.?\d+$')]
                    df[selected_col] = pd.to_numeric(df[selected_col], errors='coerce')
                    st.success(f"Dropped rows where `{selected_col}` had non-pure numbers.")

                # Save updates
                st.session_state.df = df
                update_dataset_in_db(user_id, df, st.session_state["dataset_id"])
                st.dataframe(df.head())
        else:
            st.success("No columns with mixed numeric/text patterns detected.")


        # Step 6: Encode Categorical Variables
    with st.expander("Encode Categorical Variables (Super Smart & Interactive)"):
        cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

        if cat_cols:
            st.write("### 📋 Detected Categorical Columns:")
            unique_counts = {col: df[col].nunique() for col in cat_cols}
            unique_df = pd.DataFrame({
                'Column': list(unique_counts.keys()),
                'Unique Values': list(unique_counts.values())
            })

            fig = px.bar(unique_df, x='Column', y='Unique Values', title="Unique Values per Categorical Column", text='Unique Values')
            st.plotly_chart(fig, use_container_width=True)

            high_cardinality_cols = [col for col, count in unique_counts.items() if count > 50]
            if high_cardinality_cols:
                st.warning(f"⚠️ High Cardinality Warning: These columns have more than 50 unique values: {', '.join(high_cardinality_cols)}. One-Hot Encoding may cause issues.")

            selected_cols = st.multiselect("Select columns to encode", options=cat_cols)

            if selected_cols:
                st.write("### 🧠 Auto-Suggested Encoding per Column:")
                suggestions = {}
                for col in selected_cols:
                    unique_vals = df[col].nunique()
                    suggestion = "One-Hot Encoding (recommended)" if unique_vals <= 10 else "Label Encoding (recommended)"
                    suggestions[col] = suggestion
                    st.markdown(f"**`{col}`** ➔ {suggestion}")

                encoding_choice = st.radio("Override encoding method manually?", ["Use Auto-Suggestions", "Force Label Encoding", "Force One-Hot Encoding"], key="override_choice")

                preview_col = st.selectbox("Preview a column's unique values before encoding:", selected_cols)
                if preview_col:
                    st.write(f"**🔍 Unique Values in `{preview_col}`:**")
                    st.dataframe(pd.DataFrame(df[preview_col].unique(), columns=[preview_col]))

                    st.write(f"**📊 Top 5 Most Frequent Values in `{preview_col}` (Pie Chart):**")
                    freq_series = df[preview_col].value_counts().head(5)
                    freq_df = freq_series.reset_index()
                    freq_df.columns = [preview_col, "Count"]
                    pie_fig = px.pie(freq_df, names=preview_col, values="Count", title=f"Top 5 {preview_col} Categories")
                    st.plotly_chart(pie_fig, use_container_width=True)

                if st.button("Apply Encoding", key="apply_encoding"):

                    # Create and download backup of selected columns
                    backup_df = df[selected_cols].copy()
                    backup_csv = backup_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Original Categorical Columns (Backup)",
                        data=backup_csv,
                        file_name="categorical_columns_backup.csv",
                        mime="text/csv"
                    )

                    # Perform encoding
                    for col in selected_cols:
                        if encoding_choice == "Use Auto-Suggestions":
                            method = suggestions[col]
                        elif encoding_choice == "Force Label Encoding":
                            method = "Label Encoding (forced)"
                        else:
                            method = "One-Hot Encoding (forced)"

                        st.markdown(f"**🔄 Encoding `{col}` using:** {method}")

                        if "Label" in method:
                            df[col] = df[col].astype('category')
                            mapping_dict = dict(enumerate(df[col].cat.categories))
                            reverse_mapping = {v: k for k, v in mapping_dict.items()}
                            df[col] = df[col].cat.codes

                            for orig_val, enc_val in reverse_mapping.items():
                                insert_encoding_map(st.session_state["dataset_id"], col, str(orig_val), int(enc_val))
                        else:
                            df = pd.get_dummies(df, columns=[col], drop_first=True)

                    st.success("✅ Categorical encoding completed and mappings stored!")
                    st.dataframe(df.head())
                    st.session_state.df = df
                    update_dataset_in_db(user_id, df, st.session_state["dataset_id"])
            else:
                st.warning("⚠️ Please select at least one column to encode.")
        else:
            st.success("🎯 No categorical columns to encode!")


    # Step 4: Handle Missing Values
    with st.expander("Handle Missing Values"):
        st.code("df.isnull().sum()", language="python")
        nulls = df.isnull().sum()
        st.dataframe(nulls)

        if nulls.sum() > 0:
            st.write(f"{nulls.sum()} Null values found.")

            missing_cols = df.columns[df.isnull().any()].tolist()
            all_selected = st.checkbox("Select all columns with missing values")

            if all_selected:
                selected_cols = st.multiselect(
                    "Select columns to handle missing values:",
                    missing_cols,
                    default=missing_cols
                )
            else:
                selected_cols = st.multiselect(
                    "Select columns to handle missing values:",
                    missing_cols
                )

            method = st.selectbox(
                "How do you want to handle missing values?",
                ["Fill with Mean", "Fill with Median", "Fill with 0", "Drop Rows"]
            )

            if st.button("Handle Missing Values Now"):
                if method == "Fill with Mean":
                    for col in selected_cols:
                        if df[col].dtype in [np.float64, np.int64]:  # Only fill numeric
                            df[col] = df[col].fillna(df[col].mean())
                elif method == "Fill with Median":
                    for col in selected_cols:
                        if df[col].dtype in [np.float64, np.int64]:
                            df[col] = df[col].fillna(df[col].median())
                elif method == "Fill with 0":
                    for col in selected_cols:
                        df[col] = df[col].fillna(0)
                elif method == "Drop Rows":
                    df = df.dropna(subset=selected_cols)

                st.session_state.df = df
                update_dataset_in_db(user_id, df, st.session_state['dataset_id'])
                st.success("✅ Missing values handled successfully!")
                st.dataframe(df.head())
        else:
            st.success("No missing values found!")



    # Step 5: Handle Duplicates
    with st.expander("Handle Duplicates (Smart & Interactive)"):
        st.subheader("Duplicate Rows")
        st.code("df.duplicated().sum()", language="python")
        
        # Check for duplicates in the entire dataset or selected columns
        dupes = df.duplicated().sum()
        st.write(f"🔴 **Found {dupes} duplicate rows.**")
        
        if dupes > 0:
            # Option to choose columns for duplicate detection
            st.write("### 🔧 Select columns for duplicate detection (optional):")
            columns_to_check = st.multiselect("Select columns", options=df.columns.tolist(), default=df.columns.tolist())

            # Show a preview of the duplicates
            st.write("### 📊 Preview of duplicate rows (if any):")
            duplicate_rows = df[df.duplicated(subset=columns_to_check, keep=False)]
            st.dataframe(duplicate_rows)

            # User options to handle duplicates
            remove_duplicates = st.checkbox("Remove all duplicate rows")
            keep_first = st.checkbox("Keep the first occurrence of duplicates")
            keep_last = st.checkbox("Keep the last occurrence of duplicates")

            # Apply the selected actions based on user choice
            if st.button("Apply Duplicate Handling"):
                if remove_duplicates:
                    df = df.drop_duplicates(subset=columns_to_check, keep=False)
                    st.success(f"✅ All duplicate rows removed.")
                elif keep_first:
                    df = df.drop_duplicates(subset=columns_to_check, keep='first')
                    st.success(f"✅ Kept the first occurrence of duplicates and removed others.")
                elif keep_last:
                    df = df.drop_duplicates(subset=columns_to_check, keep='last')
                    st.success(f"✅ Kept the last occurrence of duplicates and removed others.")
                else:
                    st.warning("No action selected for handling duplicates.")
                    
                st.dataframe(df.head())
                st.session_state.df = df
                update_dataset_in_db(user_id, df, st.session_state["dataset_id"])

        else:
            st.success("🎉 No duplicate rows found!")

    # Final download step
    st.markdown("---")
    st.header(" Download Final Dataset")
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("Download CSV", csv, file_name="preprocessed_dataset.csv", mime='text/csv')
