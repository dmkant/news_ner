import spacy
import pandas as pd
import numpy as npy
import json
from tqdm import tqdm
import string
import newspaper as np
import ssl
import feedparser as fp
import re
import itertools
from IPython.display import display
from copy import deepcopy
import plotly.express as px
from spacy.tokens import DocBin

from sklearn.cluster import KMeans
import wikipedia

from utils import *

def test(a):
    return a

nlp = spacy.load("fr_core_news_md")

# Initialisation
with open("data/sources.json") as mon_fichier:
    dict_sources = json.load(mon_fichier)

with open("data/lesechos.json") as file:
    dict_article = json.load(file)

for url, article in dict_article.items():
    article["source"] = "Les Echos"

df_old_article = pd.DataFrame(dict_article).T.reset_index()
df_old_article.rename(columns={"index": "url"}, inplace=True)

# Read Bytes file
with open("data/article_docs_bytes", "rb") as f:
    bytes_data_read = f.read()

# deserialize docs
doc_bin = DocBin().from_bytes(bytes_data_read)
df_old_article["docs"] = list(doc_bin.get_docs(nlp.vocab))