import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go 
from sentence_transformers import SentenceTransformer, util
import torch
import joblib

@st.cache_resource
def load_data():
    # Load from files (adjust paths as needed)
    embeddings = np.load(r"models\title_embeddings_Alibaba.npy")
    umap_2d = np.load(r"models\titles_umap_2d_AB.npy")   
    documents = pd.read_csv(r"data\full_df_minus_nan_titles.csv").title_narrative
    docs = [x.replace("['","").replace("']","").replace('["',"").replace('"]','') for x in documents if type(x)==str]
    umap_model = joblib.load(r"models\umap_2d_AB.pkl")  # use the fitted model used on training data
    pca_model = joblib.load(r"models\pca_AB.pkl")  
    return embeddings, umap_2d, docs, umap_model, pca_model

embeddings, umap_2d, documents,  umap_model, pca_model = load_data()

@st.cache_resource
def load_model():
    return SentenceTransformer('Alibaba-NLP/gte-multilingual-base', trust_remote_code=True)

model = load_model()

st.title("Semantic Search - Projects - Alibaba Embeddings")
st.markdown("Enter a query to highlight semantically similar documents on the 2D UMAP plot.")

query = st.text_input("Enter search query:")
top_k = st.slider("Number of matches to highlight", min_value=10, max_value=2500, value=100)

similarity_measure = st.radio(
    "Similarity measure",
    ["Cosine", "Euclidean", "Manhattan (L1)"],
    horizontal=True
)

if query:
    # Encode query and compute similarity
    
    with st.spinner("Searching..."):
        query_embedding = model.encode(query, convert_to_tensor=True)
        query_numpy = query_embedding.cpu().numpy().reshape(1, -1)
        query_pca = pca_model.transform(query_numpy)         # if PCA was used
        query_umap = umap_model.transform(query_pca) 
        if similarity_measure == "Cosine":
            scores = util.cos_sim(query_embedding, embeddings)[0]
        elif similarity_measure == "Euclidean":
            scores = util.euclidean_sim(query_embedding, embeddings)[0]
        elif similarity_measure == "Manhattan (L1)":
            scores = util.manhattan_sim(query_embedding, embeddings)[0]
        top_results = scores.argsort(descending=True)
        top_nk = top_results[:top_k]
        highlight_indices = top_nk.numpy()

    # Create DataFrame for plotting
    labels = ["Match" if i in highlight_indices else "Other" for i in range(len(documents))]

    df = pd.DataFrame({
        "UMAP_1": umap_2d[:, 0],
        "UMAP_2": umap_2d[:, 1],
        "Label": labels,
        "Text": documents
    })


    # Truncate text for tooltip and title
    df["Title"] = df["Text"].str.slice(0, 100) + "..."
    df["Index"] = df.index

    # Custom color mapping
    color_discrete_map = {
        "Match": "red",
        "Other": "lightgray"
    }

    # Plot
    fig = px.scatter(
        df,
        x="UMAP_1",
        y="UMAP_2",
        color="Label",
        color_discrete_map = color_discrete_map,
        hover_data={"Text": False,
                    "Title": True,
                    "Index":True,
                    "UMAP_1":False,
                    "UMAP_2":False},
        opacity=0.7,
        title=f"Top {top_k} semantic matches for: '{query}' using {similarity_measure} similarity",
        width=900,
        height=700
    )
    fig.add_trace(
        go.Scatter(
            x=[query_umap[0][0]],
            y=[query_umap[0][1]],
            mode='markers+text',
            marker=dict(size=10, color='blue', symbol='x'),
            name='Query',
            text=['Query'],
            textposition='top center',
            showlegend=True))
    fig.update_traces(marker=dict(size=4))
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Top 10 matched documents")
    for rank, (idx, score) in enumerate(zip(highlight_indices[:10], top_results[:10]), start=1):
        st.markdown(f"{rank}. {documents[idx]}")

else:
    st.info("Enter a search query to begin.")

