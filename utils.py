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


# IMPORTATION
def extract_url_first_part(url: str) -> str:
    matching = re.match(r"^(.*)\.php.*", url)
    if matching:
        return matching.group(1)
    else:
        return url


def article_already_present(df_article: pd.DataFrame, item) -> bool:
    return (
        item.link in [extract_url_first_part(url) for url in df_article["url"]]
        or item.link in df_article["url"]
    )


def ajout_article(dict_all_source: dict, df_article: pd.DataFrame) -> pd.DataFrame:
    dict_article = {}
    for source_name, dict_source in dict_all_source.items():
        for categorie, url in dict_source.items():
            print(f"Categorie '{categorie}'")
            data = fp.parse(url)
            for item in tqdm(data.entries):
                if not article_already_present(df_article, item):
                    try:
                        article = np.Article(item.link)
                        article.download()
                        article.parse()
                        dict_article[item.link] = {
                            "source": source_name,
                            "title": item.title,
                            "category": categorie,
                            "date": item.published,
                            "author": article.authors,
                            "content": article.text,
                        }
                    except:
                        pass
    df_article = pd.DataFrame(dict_article).T.reset_index()
    df_article.rename(columns={"index": "url"}, inplace=True)
    return df_article