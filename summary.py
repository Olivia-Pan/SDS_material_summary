# summary
import streamlit as st
import course1
import course2
import course3
import course4

st.set_page_config(
    page_title="Statistics and Data Science Material Summary",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Main page header ---
st.title("Statistics and Data Science Material Summary")

# --- Sidebar navigation ---
st.sidebar.title("Navigation")
unit = st.sidebar.radio(
    "Select Unit:",
    (
        "Unit 1: Intro to Data Science",
        "Unit 2: Statistical Thinking",
        "Unit 3: Probability & Inference",
        "Unit 4: Regression Methods",
    ),
)

# --- Render selected unit ---
if unit == "Unit 1: Intro to Data Science":
    course1.show()
elif unit == "Unit 2: Statistical Thinking":
    course2.show()
elif unit == "Unit 3: Probability & Inference":
    course3.show()
elif unit == "Unit 4: Regression Methods":
    course4.show()
