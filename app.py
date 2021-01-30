# coding: utf-8
"""
Example of a Streamlit app for an interactive spaCy model visualizer.
Usage:
streamlit run app.py
"""
from __future__ import unicode_literals
import os
import sys

import streamlit as st
import spacy
from spacy import displacy
import pandas as pd

path_this = os.path.dirname(os.path.abspath(__file__))
path_model = os.path.abspath(os.path.join(path_this, 'model'))
print (path_model)

sys.path.append(path_this)
sys.path.append(path_model)

SPACY_MODEL_NAMES = ["id_maslahah_ner", "id_maslahah_sentiment"]
HTML_WRAPPER = """<div style="overflow-x: auto; border: 1px solid #e6e9ef; border-radius: 0.25rem; padding: 1rem; margin-bottom: 2.5rem">{}</div>"""


@st.cache(allow_output_mutation=True)
def load_model(name):
    return spacy.load(name)


@st.cache(allow_output_mutation=True)
def process_text(model_name, text):
    nlp = load_model(os.path.join(path_model, model_name))
    return nlp(text)


st.sidebar.title("Maslahah-Interactive NER&Sentiment Visualizer")
st.sidebar.markdown("""Proses teks menggunakan [spaCy](https://spacy.io) models untuk memvisualisasikan NER dan klasifikasi teks."""
)

spacy_model = st.sidebar.selectbox("Model name", SPACY_MODEL_NAMES)
model_load_state = st.info(f"Loading model '{spacy_model}'...")
if spacy_model == "id_maslahah_ner":
    DEFAULT_TEXT = "Joko Widodo adalah presiden dari Partai PDI-Perjuangan. Beliau beristana di Jakarta."
elif spacy_model == "id_maslahah_sentiment":
    DEFAULT_TEXT = "Aku suka banget sama ini. Cuma ini cleanser yang cocok buat aku. Aku udah sering banget gonta-ganti cleanser karena wajahku yang penuh jerawat. Aku udah coba yang low-end dan high-end sekalipun tapi tetep gak ada yang sebagus ini. Karena produknya gentle jadi gak bikin iritasi diwajah, kebanyakan produk cleanser lain yang harsh untuk wajah justru dapat menyebabkan iritasi sehingga munculah jerawat. Very recommended!"
nlp = load_model(os.path.join(path_model, spacy_model))
model_load_state.empty()

text = st.text_area("Text to analyze", DEFAULT_TEXT)
doc = process_text(spacy_model, text)

if "ner" in nlp.pipe_names:
    st.header("Named Entities")
    st.sidebar.header("Named Entities")
    label_set = nlp.get_pipe("ner").labels
    labels = st.sidebar.multiselect(
        "Entity labels", options=label_set, default=list(label_set)
    )
    html = displacy.render(doc, style="ent", options={"ents": labels})
    # Newlines seem to mess with the rendering
    html = html.replace("\n", " ")
    st.write(HTML_WRAPPER.format(html), unsafe_allow_html=True)
    attrs = ["text", "label_", "start", "end", "start_char", "end_char"]
    if "entity_linker" in nlp.pipe_names:
        attrs.append("kb_id_")
    data = [
        [str(getattr(ent, attr)) for attr in attrs]
        for ent in doc.ents
        if ent.label_ in labels
    ]
    df = pd.DataFrame(data, columns=attrs)
    st.dataframe(df)

if "textcat" in nlp.pipe_names:
    st.header("Text Classification")
    st.markdown(f"> {text}")
    df = pd.DataFrame(doc.cats.items(), columns=("Label", "Score"))
    st.dataframe(df)

st.header("JSON Doc")
if st.button("Show JSON Doc"):
    st.json(doc.to_json())

st.header("JSON model meta")
if st.button("Show JSON model meta"):
    st.json(nlp.meta)