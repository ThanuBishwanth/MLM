# pages/home.py
import streamlit as st
import psycopg2
import bcrypt
from database.db import get_connection
import importlib
def render():
    st.title(" MLMCreator")
    st.subheader("End-to-end machine learning model creation made easy.")

    if st.session_state.get("authenticated"):
        st.text("Already Logged in, Please carry on")
        return
    
    if "show_register" not in st.session_state:
        st.session_state["show_register"] = False

    if st.session_state["show_register"]:
        render_register()
    else:
        render_login()

def render_login():
    st.header("Login")
    email = st.text_input("Email")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if not email or not password:
            st.error("Please fill all fields.")
            return

        conn = get_connection()
        cur = conn.cursor()
        cur.execute("SELECT id, password FROM users WHERE email = %s", (email,))
        user = cur.fetchone()
        cur.close()
        conn.close()

        if user and bcrypt.checkpw(password.encode(), user[1].encode()):
            st.session_state["authenticated"] = True
            st.session_state["user_id"] = user[0]
            st.session_state.df=None
            st.session_state['dataset_id']=None
            st.success("Login successful.")
        else:
            st.error("Invalid credentials.")

    if st.button("Register New Account"):
        st.session_state["show_register"] = True

def render_register():
    st.header("Register")
    email = st.text_input("Email")
    password = st.text_input("Create Password", type="password")

    if st.button("Create Account"):
        if not email or not password:
            st.error("Please fill all fields.")
            return

        hashed = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
        try:
            conn = get_connection()
            cur = conn.cursor()
            cur.execute("INSERT INTO users (email, password) VALUES (%s, %s)", (email, hashed))
            conn.commit()
            cur.close()
            conn.close()
            st.success("Account created. Please login.")
            st.session_state["show_register"] = False
        except psycopg2.Error:
            st.error("Email already registered.")

    if st.button("Back to Login"):
        st.session_state["show_register"] = False
