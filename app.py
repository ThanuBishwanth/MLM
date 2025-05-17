# app.py
import streamlit as st
from streamlit_option_menu import option_menu
import time
import importlib

st.set_page_config(page_title="MLMCreator",layout="wide")

# Page order and routing
PAGE_TITLES = {
    "Home": "page.home",
    "Upload": "page.upload",
    "Pre-Process": "page.preprocess",
    "Train": "page.train",
    "Models": "page.models",
    "Predict":"page.predict",
    "Clustering":"page.clustering"
}

# Session setup
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False
if "user_id" not in st.session_state:
    st.session_state["user_id"] = None
if "last_activity" not in st.session_state:
    st.session_state["last_activity"] = time.time()

# Timeout logic (10 min = 600 seconds)
if st.session_state["authenticated"]:
    if time.time() - st.session_state["last_activity"] > 1800:
        st.session_state["authenticated"] = False
        st.session_state["user_id"] = None
        st.warning("Session expired. Please log in again.")
        st.stop()
    else:
        st.session_state["last_activity"] = time.time()

# Sidebar Navigation
with st.sidebar:
    st.session_state.selected = option_menu(
        menu_title=None,
        options=["Home", "Upload", "Pre-Process","Clustering", "Train", "Models","Predict"],
        icons=["house", "cloud-upload", "gear","share", "bar-chart", "folder",""],
        default_index=0
    )
    st.markdown("---")
    if st.button("Logout"):
        st.session_state["authenticated"] = False
        st.session_state["user_id"] = None
        st.rerun()

# Render selected page
if st.session_state.selected in PAGE_TITLES:
    module = importlib.import_module(PAGE_TITLES[st.session_state.selected])
    module.render()
