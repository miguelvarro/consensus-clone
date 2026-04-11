import streamlit as st
import requests

st.title("AI Consensus Clone (MVP)")

api_url = st.text_input("API URL", "http://localhost:8000")
q = st.text_input("Consulta")
k = st.slider("Top-K", 3, 20, 8)

if st.button("Buscar"):
    r = requests.post(f"{api_url}/search", json={"q": q, "k": k})
    st.json(r.json())

if st.button("Responder"):
    r = requests.post(f"{api_url}/answer", json={"q": q, "k": k})
    st.json(r.json())

