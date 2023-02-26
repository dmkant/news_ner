import streamlit as st
from preprocess import *
import spacy

def get_html(html: str):
    """Convert HTML so it can be rendered."""
    WRAPPER = """<div style="overflow-x: auto; border: 1px solid #e6e9ef; border-radius: 0.25rem; padding: 1rem; margin-bottom: 2.5rem">{}</div>"""
    # Newlines seem to mess with the rendering
    html = html.replace("\n", " ")
    return WRAPPER.format(html)

nlp = spacy.load("fr_core_news_md")

st.set_page_config(layout="wide")
# st.markdown("<h1 style='text-align: center; color: red;'>Analyse des Entités Nommées dans la presse</h1>", unsafe_allow_html=True)


st.title("Analyse des Entités Nommées dans la presse")
st.subheader("Recognition and Relation Extraction")

################### SideBar
st.sidebar.text(test("ok"))

source_filter = st.sidebar.selectbox(
    "Filter la source",
    ("ALL", "Les Echos")
)

doc = nlp("Jean Marc est CTO en Espagne et Covid")
html = spacy.displacy.render( doc,style="ent")
style = "<style>mark.entity { display: inline-block }</style>"
st.write(f"{style}{get_html(html)}", unsafe_allow_html=True)