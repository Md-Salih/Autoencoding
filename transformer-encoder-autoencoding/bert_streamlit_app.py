import streamlit as st
from transformers import pipeline

st.title("BERT Masked Language Model Demo")
st.write("Enter a sentence with a [MASK] token. The model will predict the masked word using pre-trained BERT.")

user_input = st.text_input("Input Sentence", "Transformers use [MASK] attention")

if st.button("Predict"): 
    nlp = pipeline('fill-mask', model='bert-base-uncased')
    results = nlp(user_input)
    st.write("### Top Predictions:")
    for res in results:
        st.success(f"{res['sequence']} (score: {res['score']:.4f})")
