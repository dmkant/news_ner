import streamlit as st
from utils import *
from preprocess import *
import spacy
import os
from streamlit_agraph import agraph, Node, Edge, Config


st.set_page_config(layout="wide")

nlp = spacy.load("fr_core_news_md")

@st.cache_resource
def fetch_data():
    # Initialisation
    with open("data/sources.json") as mon_fichier:
        dict_sources = json.load(mon_fichier)

    with open("data/lesechos.json") as file:
        dict_article = json.load(file)

    df_old_article = pd.DataFrame(dict_article).T.reset_index()
    df_old_article.rename(columns={"index": "url"}, inplace=True)

    # Read Bytes file
    with open("data/article_docs_bytes", "rb") as f:
        bytes_data_read = f.read()

    # deserialize docs
    doc_bin = DocBin().from_bytes(bytes_data_read)
    df_old_article["docs"] = list(doc_bin.get_docs(nlp.vocab))

    # Ajout nouveaux article
    df_new_article = ajout_article(dict_all_source=dict_sources, df_article=df_old_article)
    res_pipe = nlp.pipe(df_new_article["content"])
    df_new_article["docs"] = [doc for doc in res_pipe]

    # Concat
    df_newspaper_article = pd.concat([df_old_article, df_new_article])
    # Detection entite
    df_newspaper_article["entity"] = df_newspaper_article["docs"].apply(lambda x: x.ents)
    
    return df_newspaper_article

df_newspaper_article = fetch_data()

# st.markdown("<h1 style='text-align: center; color: red;'>Analyse des Entités Nommées dans la presse</h1>", unsafe_allow_html=True)


st.title("Analyse des Entités Nommées dans la presse")
st.subheader("Recognition and Relation Extraction")

################################################################################################# SideBar
st.sidebar.title("Paramètres")

source_filter = st.sidebar.selectbox(
    "Filter la source",
    ["ALL"] + list(df_newspaper_article["source"].unique())
)

category_filter = st.sidebar.selectbox(
    "Filter la catégory de l'article",
    ["ALL"] + list(df_newspaper_article["category"].unique())
)

nb_max_entite_filter = st.sidebar.number_input("Nombre maximum d'entités",min_value=5,step=1)


type_entite1_filter = st.sidebar.selectbox(
    "Filter le type du la première entité",
    ["ALL","PER","LOC","ORG"]
)
type_entite2_filter = st.sidebar.selectbox(
    "Filter le type du la deuxième entité",
    ["ALL","PER","LOC","ORG"]
)
nb_max_relation_filter = st.sidebar.number_input("Nombre maximum de relations issues de la presse",value=50,min_value=5,step=1)
nb_max_relation_wiki_filter = st.sidebar.number_input("Nombre maximum de relations issues Wikipedia",value=20,min_value=1,step=1)
nb_cluster =  st.sidebar.number_input("Nombre de cluster regroupant les relations entre entités similaires",value=4,min_value=2,step=1)

#######################################################################################################

df_article = filter_article(
    df_all_article=df_newspaper_article,
    source_filter=source_filter,
    category_filter=category_filter,
)

top_container = st.container()
top_col1,top_col2 = top_container.columns([1,3])
top_col1.metric("Nombre d'articles", f"{df_article.shape[0]}")


top_col1.write("Sources des article:")
for img_file in os.listdir("data/img/sources"):
    top_col1.image(f"data/img/sources/{img_file}",use_column_width="always")
    top_col1.image(f"data/img/sources/{img_file}",use_column_width="always")



df_main_entity = get_main_entity(
    N_most_common=nb_max_entite_filter, df_article=df_article
)
fig = px.bar(
    df_main_entity.iloc[::-1].rename(columns={"lemma": "entité"}),
    y="entité",
    x="occurence",
    color="type",
    width=1000,
    height=900,
    title="Occurence des entités dans les articles selon leur type",
)
top_col2.plotly_chart(fig,use_container_width=True)


@st.cache_resource
def add_wikipedia(df_main_entity):
    df_wikipedia_article = ajout_wikipedia_article(df_main_entity=df_main_entity)
    # Application de la pipeline
    res_pipe = nlp.pipe(df_wikipedia_article["content"])
    df_wikipedia_article["docs"] = [doc for doc in res_pipe]
    df_wikipedia_article["entity"] = df_wikipedia_article["docs"].apply(lambda x: x.ents)
    
    return df_wikipedia_article

df_wikipedia_article = add_wikipedia(df_main_entity=df_main_entity[["type","lemma"]]) 
df_article = pd.concat([df_article, df_wikipedia_article])

# Sent entity
df_sent_entity = get_df_sent_entity(df_article=df_article)

# Relations Journeaux
# Get Entity
df_entity_relation_newspaper = get_entity_relation(
    df_sent_entity=df_sent_entity[df_sent_entity["source"] != "Wikipedia"],
    type_entite1_filter=type_entite1_filter,
    type_entite2_filter=type_entite2_filter,
)

df_entity_relation_newspaper = df_entity_relation_newspaper[
    (df_entity_relation_newspaper["entite1"].isin(df_main_entity["lemma"]))
    & (df_entity_relation_newspaper["entite2"].isin(df_main_entity["lemma"]))
]

# Filter
df_entity_relation_filter = (
    df_entity_relation_newspaper.groupby(["type1", "entite1", "relation", "entite2", "type2"])
    .aggregate(**{"num_phrase": ("num_phrase", list), "occurence": ("num_phrase", len)})
    .reset_index()
    .sort_values("occurence", ascending=False,ignore_index=True)
    .head(nb_max_relation_filter)
)

#Relation Wikipideia
df_entity_relation_wiki = get_entity_relation(
    df_sent_entity=df_sent_entity[df_sent_entity["source"] == "Wikipedia"],
    type_entite1_filter=type_entite1_filter,
    type_entite2_filter=type_entite2_filter,
)

df_entity_relation_wiki = df_entity_relation_wiki[(df_entity_relation_wiki["entite1"].isin(df_entity_relation_filter["entite1"])) 
                                                  | (df_entity_relation_wiki["entite1"].isin(df_entity_relation_filter["entite2"]))
                                                  | (df_entity_relation_wiki["entite2"].isin(df_entity_relation_filter["entite1"])) 
                                                  | (df_entity_relation_wiki["entite2"].isin(df_entity_relation_filter["entite2"]))]
# Filter
df_entity_relation_wiki_filter = (
    df_entity_relation_wiki.groupby(["type1", "entite1", "relation", "entite2", "type2"])
    .aggregate(**{"num_phrase": ("num_phrase", list), "occurence": ("num_phrase", len)})
    .reset_index()
    .sort_values("occurence", ascending=False,ignore_index=True)
    .head(nb_max_relation_wiki_filter)
)

# Concat

df_entity_relation_filter = pd.concat([df_entity_relation_filter.assign(source="paper"),
                                       df_entity_relation_wiki_filter.assign(source="Wikipedia")])


# Clustering
embedding_relation = df_entity_relation_filter["relation"].apply(nlp.vocab.get_vector)
embedding_relation = npy.array([array for array in embedding_relation],dtype=float).T
kmeans = KMeans(n_clusters=min(nb_cluster,df_entity_relation_filter.shape[0]), random_state=0, n_init="auto").fit(npy.transpose(embedding_relation))

df_entity_relation_filter["cluster"] = kmeans.labels_
df_entity_relation_filter = df_entity_relation_filter.drop_duplicates(subset=["entite1","entite2"])

######################################### Graphe

st.cache_resource
def graph(df_entity_relation_filter):
    nodes = []
    edges = []
    idnodes = []
    for i in range(df_entity_relation_filter.shape[0]):
        if df_entity_relation_filter["entite1"].iloc[i] not in idnodes:
            nodes.append( Node(id=df_entity_relation_filter["entite1"].iloc[i], 
                        label=df_entity_relation_filter["entite1"].iloc[i], 
                        size=15, 
                        shape="square",
                        color = "red") 
                    )
            idnodes.append(df_entity_relation_filter["entite1"].iloc[i] )
        
        if df_entity_relation_filter["entite2"].iloc[i] not in idnodes:
            nodes.append( Node(id=df_entity_relation_filter["entite2"].iloc[i], 
                        label=df_entity_relation_filter["entite2"].iloc[i], 
                        size=15, 
                        shape="circus",
                        color = "red") 
                    )
            idnodes.append(df_entity_relation_filter["entite2"].iloc[i] )
        
        
        edges.append( Edge(
            source=df_entity_relation_filter["entite1"].iloc[i],
            label=df_entity_relation_filter["relation"].iloc[i], 
            target=df_entity_relation_filter["entite2"].iloc[i], 
                        # **kwargs
                        ) 
                    ) 

    config = Config(height=900, width=900, nodeHighlightBehavior=False, highlightColor="#F7A7A6", directed=False,collapsible=False)

    agraph(nodes=nodes, 
           edges=edges, 
           config=config)
    
graph(df_entity_relation_filter)

@st.cache_resource
def display_sentance(df_entity_relation_filter):
    # Affiche les relations dans leur context
    for i in df_entity_relation_filter["num_phrase"].iloc[0]:
        html = spacy.displacy.render(df_sent_entity.loc[i,"sent"], style="ent")
        style = "<style>mark.entity { display: inline-block }</style>"
        st.write(f"{style}{get_html(html)}", unsafe_allow_html=True)
        
        st.write(df_sent_entity.loc[i,["source","url"]])



