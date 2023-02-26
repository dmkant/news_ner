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
import streamlit as st
from sklearn.cluster import KMeans
import wikipedia
import colorsys

def get_html(html: str):
    """Convert HTML so it can be rendered."""
    WRAPPER = """<div style="overflow-x: auto; border: 1px solid #e6e9ef; border-radius: 0.25rem; padding: 1rem; margin-bottom: 2.5rem">{}</div>"""
    # Newlines seem to mess with the rendering
    html = html.replace("\n", " ")
    return WRAPPER.format(html)


def get_N_HexCol(N=5):
    HSV_tuples = [(x * 1.0 / N, 0.5, 0.5) for x in range(N)]
    hex_out = []
    for rgb in HSV_tuples:
        rgb = map(lambda x: int(x * 255), colorsys.hsv_to_rgb(*rgb))
        hex_out.append('#%02x%02x%02x' % tuple(rgb))
    return hex_out

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

def filter_article(
    df_all_article: pd.DataFrame, source_filter: str, category_filter: str
) -> pd.DataFrame:
    # filtrer
    source_filter = (df_all_article["source"] == source_filter) & (source_filter != "ALL") | (source_filter == "ALL")
    category_filter = (df_all_article["category"] == category_filter) & (category_filter != "ALL") | (category_filter == "ALL")
    df_article = df_all_article[(source_filter) & (category_filter)].reset_index(drop=True)

    return df_article

############ Get entities
def get_main_entity(
    df_article: pd.DataFrame ,N_most_common: int = 20
) -> pd.DataFrame:
    df_entity = pd.DataFrame(
        (
            df_article[df_article["entity"].apply(len) > 0]
            .apply(
                lambda x: [
                    (x["url"], ent, ent.label_, ent.lemma_) for ent in x["entity"]
                ],
                axis=1,
            )
            .explode()
            .tolist()
        ),
        columns=["url", "entity", "type", "lemma"],
    )

    df_main_entity = (
        df_entity[["type", "lemma"]]
        .value_counts()
        .reset_index(name="occurence")
        .groupby("type",group_keys=True)[["lemma", "occurence"]]
        .apply(lambda grp: grp.nlargest(N_most_common, "occurence"))
        .reset_index()
    )

    return df_main_entity


###################### Add wikipedia 
def ajout_wikipedia_article(df_main_entity: pd.DataFrame) -> pd.DataFrame:
    list_article = []
    wikipedia.set_lang("fr")

    for entite_lemma in tqdm(df_main_entity[df_main_entity["type"]!="MISC"]["lemma"]):
        try:
            wiki_search = wikipedia.search(entite_lemma,results=1)
            if len(wiki_search) > 0 :
                wiki_page = wikipedia.WikipediaPage(title = wiki_search[0])
                list_article.append({
                    "url":wiki_page.url,
                    "source": "Wikipedia",
                    "title": wiki_page.title,
                    "content": wiki_page.summary,
                })
        except:
            pass
    df_article = pd.DataFrame(list_article)
    return df_article

###################### sentences entities
def get_df_sent_entity(df_article: pd.DataFrame) -> pd.DataFrame:
    list_sent = [
        {
            "num_doc": df_article.index[num_doc],
            "source": df_article["source"].iloc[num_doc],
            "url": df_article["url"].iloc[num_doc],
            "sent": sent,
            "entity": sent.ents,
            "entity_lemma": [ent.lemma_ for ent in sent.ents],
        }
        for num_doc, doc in enumerate(df_article["docs"])
        for sent in doc.sents
        if len(sent.ents) >= 2
    ]
    
    df = pd.DataFrame(list_sent)
    return df

####################### Entity relation
def relation_common_head(entity1, entity2):
    same_head = entity1.root.head == entity2.root.head
    not_root = entity1.root.dep_ != "ROOT" and entity2.root.dep_ != "ROOT"
    if same_head and not_root:
        return entity1.root.head
    else:
        return None

def filter_entity_relation(
    entity1, entity2, type_entite1_filter: str, type_entite2_filter: str
) -> bool:
    not_same_lemma = entity1.lemma_ != entity2.lemma_
    if type_entite1_filter == "ALL" and type_entite2_filter == "ALL":
        return not_same_lemma

    elif type_entite1_filter != "ALL" and type_entite2_filter == "ALL":
        good_type = (
            entity1.label_ == type_entite1_filter
            or entity2.label_ == type_entite1_filter
        )

    elif type_entite1_filter == "ALL" and type_entite2_filter != "ALL":
        good_type = (
            entity1.label_ == type_entite2_filter
            or entity2.label_ == type_entite2_filter
        )

    else:
        good_type1 = (
            entity1.label_ == type_entite1_filter
            and entity2.label_ == type_entite2_filter
        )
        good_type2 = (
            entity1.label_ == type_entite2_filter
            and entity2.label_ == type_entite1_filter
        )
        good_type = good_type1 or good_type2

    return not_same_lemma and good_type

def get_entity_relation(
    df_sent_entity: pd.DataFrame, type_entite1_filter: str, type_entite2_filter: str
) -> pd.DataFrame:
    list_entity_relation = []

    for i in tqdm(range(df_sent_entity.shape[0])):
        for entity1, entity2 in itertools.combinations(df_sent_entity["entity"].iloc[i], 2):
            # Ordonne alphabetiquement
            entity1, entity2 = [entity for _,entity in sorted(zip([entity1.lemma_,entity2.lemma_],[entity1, entity2]))]
            relevant_relation = filter_entity_relation(
                entity1=entity1,
                entity2=entity2,
                type_entite1_filter=type_entite1_filter,
                type_entite2_filter=type_entite2_filter,
            )
            if relevant_relation:
                common_head_relation = relation_common_head(entity1, entity2)
                if common_head_relation is not None:
                    list_entity_relation.append(
                        {
                            "source": df_sent_entity["source"].iloc[i],
                            "url": df_sent_entity["url"].iloc[i],
                            "type1": entity1.label_,
                            "entite1": entity1.lemma_,
                            "relation": common_head_relation.lemma_,
                            "entite2": entity2.lemma_,
                            "type2": entity2.label_,
                            "num_phrase": df_sent_entity["url"].index[i],
                        }
                    )

    df_entity_relation = pd.DataFrame(list_entity_relation)

    return df_entity_relation