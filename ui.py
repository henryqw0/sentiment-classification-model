import streamlit as st
import pickle
import time
# Load Model
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
# Title
st.title("SENTIMENT CLASSIFIER\n This classifier determines if a text is positive, neutral or negative. It is well suited for both short and long texts")

# Initialization and Setup
if "text" not in st.session_state:
    st.session_state.text = ""

if "checked" not in st.session_state:
    st.session_state.checked = False

def check():
    st.session_state.checked = True

def clear():
    st.session_state.text = ""
    st.session_state.checked = False
# User Input
user_input = st.text_area(label="", placeholder="Enter the text to classify", key="text")
# User Interaction
if st.button("CHECK", on_click=check):
     with st.spinner("Processing...", show_time=True):
                   time.sleep(2)
st.button("CLEAR TEXT", on_click=clear)
# User Input validation and Result
if st.session_state.checked:
    if user_input.strip() == "":
        st.warning("Enter text first")
    else:
        vec = vectorizer.transform([user_input])
        pred = model.predict(vec)[0]
        proba = model.predict_proba(vec)[0]
        
        labels = { 0: "Negative", 1: "Neutral", 2: "Positive" }
        confidence = max(proba)
        if vec.nnz == 0 or confidence < 0.5:
                st.write("Prediction(low confidence):" , labels[pred])
                st.write(f"Confidence:  {confidence:.2f}")
                st.write(f"Reason: Texts contains unfamiliar words or weak signals.")
        else:
                st.write("Prediction:" , labels[pred])
                st.write(f"Confidence:  {confidence:.2f}")


    