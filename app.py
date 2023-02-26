import streamlit as st
from utils import *
import spacy
import os
from streamlit_agraph import agraph, Node, Edge, Config

ENT_COLOR = {"PER":"#4D9DE0","LOC":"#E15554","ORG":"#E1BC29","MISC":"#3BB273"}

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

st.markdown("<h1 style='text-align: center; color: red;'>Analyse des Entités Nommées dans la presse</h1>", unsafe_allow_html=True)
st.subheader("Reconnaissance des Entités Nommées")

################################################################################################# SideBar
st.sidebar.title("Paramètres")
st.sidebar.markdown("<h4>Filter:</h4>",unsafe_allow_html=True)
top_container = st.sidebar.container()
top_col1,top_col2 = top_container.columns([1,1])
source_filter = top_col1.selectbox(
    "La source",
    ["ALL"] + list(df_newspaper_article["source"].unique())
)

category_filter = top_col2.selectbox(
    "La catégory de l'article",
    ["ALL"] + list(df_newspaper_article["category"].unique())
)

nb_max_entite_filter = st.sidebar.number_input("Nombre maximum d'entités",value=20,min_value=5,step=1)

st.sidebar.markdown("<hr></hr>", unsafe_allow_html=True)
st.sidebar.markdown("<h4>Filter le type de:</h4>",unsafe_allow_html=True)

top_container = st.sidebar.container()
top_col1,top_col2 = top_container.columns([1,1])
type_entite1_filter = top_col1.selectbox(
    "La première entité",
    ["ALL","PER","LOC","ORG"]
)
type_entite2_filter = top_col2.selectbox(
    "La seconde entité",
    ["ALL","PER","LOC","ORG"]
)

st.sidebar.markdown("<h4>Nombre maximum de relations issues de:</h4>",unsafe_allow_html=True)
top_container = st.sidebar.container()
top_col1,top_col2 = top_container.columns([1,1])
nb_max_relation_filter = top_col1.number_input("Les articles",value=50,min_value=5,step=1)
nb_max_relation_wiki_filter = top_col2.number_input("Wikipedia",value=20,min_value=1,step=1)
nb_cluster =  st.sidebar.number_input("Nombre de cluster regroupant les relations entre entités similaires",value=4,min_value=2,step=1)

#######################################################################################################

@st.cache_resource
def display_main_kpi(source_filter,category_filter,nb_max_entite_filter):
    df_article = filter_article(
        df_all_article=df_newspaper_article,
        source_filter=source_filter,
        category_filter=category_filter,
    )

    top_container = st.container()
    top_col1,top_col2 = top_container.columns([2,5])
    top_col1.markdown(f"<h4> Nombre d'articles</h4> <h2 style='color: red;'> {df_article.shape[0]} <h/2>",unsafe_allow_html=True)


    top_col1.write("Sources des articles:")
    for img_file in os.listdir("data/img/sources"):
        top_col1.image(f"data/img/sources/{img_file}",use_column_width="always")
        top_col1.image(f"data/img/sources/{img_file}",use_column_width="always")
        top_col1.image(f"data/img/sources/{img_file}",use_column_width="always")
        top_col1.image(f"data/img/sources/{img_file}",use_column_width="always")



    df_main_entity = get_main_entity( N_most_common=nb_max_entite_filter, df_article=df_article)
    fig = px.bar(
        df_main_entity.iloc[::-1].rename(columns={"lemma": "entité"}),
        y="entité",
        x="occurence",
        color="type",
        color_discrete_map=ENT_COLOR,
        width=1000,
        height=900,
        title="Occurence des entités dans les articles selon leur type",
    )
    top_col2.plotly_chart(fig,use_container_width=True)



    df_wikipedia_article = ajout_wikipedia_article(df_main_entity=df_main_entity)
    # Application de la pipeline
    res_pipe = nlp.pipe(df_wikipedia_article["content"])
    df_wikipedia_article["docs"] = [doc for doc in res_pipe]
    df_wikipedia_article["entity"] = df_wikipedia_article["docs"].apply(lambda x: x.ents)
    df_article = pd.concat([df_article, df_wikipedia_article])

    # Sent entity
    df_sent_entity = get_df_sent_entity(df_article=df_article)
    
    return df_main_entity,df_sent_entity

df_main_entity,df_sent_entity = display_main_kpi(source_filter,category_filter,nb_max_entite_filter)

@st.cache_resource
def display_entite(type_entite1_filter,type_entite2_filter,nb_max_relation_filter,nb_max_relation_wiki_filter,nb_cluster):
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
    
    df_entity_relation_filter = pd.concat([df_entity_relation_filter.assign(source="paper"),
                                           df_entity_relation_wiki_filter.assign(source="Wikipedia")])

    # Clustering
    embedding_relation = df_entity_relation_filter["relation"].apply(nlp.vocab.get_vector)
    embedding_relation = npy.array([array for array in embedding_relation],dtype=float).T
    kmeans = KMeans(n_clusters=min(nb_cluster,df_entity_relation_filter.shape[0]), random_state=0, n_init="auto").fit(npy.transpose(embedding_relation))

    df_entity_relation_filter["cluster"] = kmeans.labels_
    color_cluster = get_N_HexCol(nb_cluster)
    df_entity_relation_filter["color_cluster"] = df_entity_relation_filter["cluster"].map({i:color_cluster[i] for i in range(nb_cluster)} )


    df_entity_relation_filter["color1"] = df_entity_relation_filter["type1"].map(ENT_COLOR)
    df_entity_relation_filter["color2"] = df_entity_relation_filter["type2"].map(ENT_COLOR)
    df_entity_relation_filter = df_entity_relation_filter.drop_duplicates(subset=["entite1","entite2"])
    
    return df_entity_relation_filter


df_entity_relation_filter = display_entite(type_entite1_filter,type_entite2_filter,nb_max_relation_filter,nb_max_relation_wiki_filter,nb_cluster)


######################################### Graphe

st.subheader("Extraction des Relations")


nodes = []
edges = []
idnodes = []
for i in range(df_entity_relation_filter.shape[0]):
    if df_entity_relation_filter["entite1"].iloc[i] not in idnodes:
        nodes.append( Node(id=df_entity_relation_filter["entite1"].iloc[i], 
                    label=df_entity_relation_filter["entite1"].iloc[i], 
                    size=15, 
                    shape="square",
                    color = df_entity_relation_filter["color1"].iloc[i]) 
                )
        idnodes.append(df_entity_relation_filter["entite1"].iloc[i] )
    
    if df_entity_relation_filter["entite2"].iloc[i] not in idnodes:
        nodes.append( Node(id=df_entity_relation_filter["entite2"].iloc[i], 
                    label=df_entity_relation_filter["entite2"].iloc[i], 
                    size=15, 
                    shape="square",
                    color = df_entity_relation_filter["color2"].iloc[i]) 
                )
        idnodes.append(df_entity_relation_filter["entite2"].iloc[i] )
    
    edge_dashed = True if df_entity_relation_filter["source"].iloc[i] == "Wikipedia" else False
    edge_width =  2+5*(df_entity_relation_filter["occurence"].iloc[i]-df_entity_relation_filter["occurence"].min())/(df_entity_relation_filter["occurence"].max()-df_entity_relation_filter["occurence"].min())
    edges.append( Edge(
        source=df_entity_relation_filter["entite1"].iloc[i],
        label=df_entity_relation_filter["relation"].iloc[i], 
        target=df_entity_relation_filter["entite2"].iloc[i], 
        dashes = edge_dashed,
        width = edge_width,
        color = df_entity_relation_filter["color_cluster"].iloc[i]
                    # **kwargs
                    ) 
                ) 

config = Config(height=900, width=1000, nodeHighlightBehavior=True, highlightColor="#F7A7A6", directed=False,collapsible=True)

print("wsh")
resu_graph = agraph(nodes=nodes, 
        edges=edges, 
        config=config)


if resu_graph:
    with st.expander("",expanded=True):
        st.subheader(f"Sources de l'extraction des relations liées à l'entité '{resu_graph}'")
        for list_num_phrase in df_entity_relation_filter.loc[(df_entity_relation_filter["entite1"]==resu_graph) | (df_entity_relation_filter["entite2"]==resu_graph),"num_phrase"]:
            for i in list_num_phrase:
                html = spacy.displacy.render(df_sent_entity.loc[i,"sent"], style="ent", options={ "colors": ENT_COLOR})
                style = "<style>mark.entity { display: inline-block }</style>"
                st.write(f"{style}{get_html(html)}", unsafe_allow_html=True)
                st.markdown(f"{df_sent_entity.loc[i,'source']}: {df_sent_entity.loc[i,'url']}", unsafe_allow_html=True)



