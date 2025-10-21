import os
import numpy as np
import pandas as pd
from pathlib import Path
import streamlit as st
import joblib
import plotly.express as px
import plotly.graph_objects as go 
import gdown
import torch
from sentence_transformers import SentenceTransformer, util

# -------------------------------
# Setup
# -------------------------------
st.set_page_config(page_title="Alibaba Semantic Search", layout="wide")

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

embeddings_path = MODEL_DIR / 'desc_embeddings_Alibaba_20251016.npy' 
umap_embeddings_path = MODEL_DIR / 'descs_umap_2d_AB_20251016.npy' 
data_file_path = MODEL_DIR / 'full_df_minus_nan_descs.csv' 
umap_model_path = MODEL_DIR / 'umap_2d_AB_written.pkl' 
pca_model_path = MODEL_DIR / 'pca_AB_written.pkl' 

emb_ID = '1QQ_QfFTSzTLNkp6Sr4jux_ZTJjMhSyah'
umap_emb_ID = '1a5t5iWOAVgUmYXzrWXctATkDyx9rRF4F'
data_ID = '1tzM67Lg3R-rAvRtol0VGHx6zGW_tdx60'
umap_mod_ID = '1x8PK1Gn72YYBZ4po-0guZMUBtL8oSn1i'
pca_mod_ID = '1jIxBBAZOy8OAzGxBCG4jy7244Wb_TjP9'

paths = [embeddings_path, umap_embeddings_path, data_file_path, umap_model_path, pca_model_path]
ids = [emb_ID, umap_emb_ID, data_ID, umap_mod_ID, pca_mod_ID]
assets_links = [f"https://drive.google.com/uc?id={x}" for x in ids]

# -------------------------------
# Download + Load Data
# -------------------------------
def load_assets():
    st.info("Downloading assets from Google Drive (only if missing)...")
    for url, path in zip(assets_links, paths):
        if not path.exists():
            gdown.download(url, str(path), quiet=False)
    st.success("Assets ready ✅")

    embeddings = np.load(embeddings_path)
    umap_2d = np.load(umap_embeddings_path)
    docs = pd.read_csv(data_file_path)
    umap_model = joblib.load(umap_model_path)
    pca_model = joblib.load(pca_model_path)
    return embeddings, umap_2d, docs, umap_model, pca_model

embeddings, umap_2d, docs, umap_model, pca_model = load_assets()

# -------------------------------
# Load SentenceTransformer (cached)
# -------------------------------
@st.cache_resource
def load_text_encoder():
    return SentenceTransformer('Alibaba-NLP/gte-multilingual-base', trust_remote_code=True)

model = load_text_encoder()

# -------------------------------
# UI
# -------------------------------
st.title("🔍 Semantic Search — Alibaba Embeddings")
st.markdown("Enter a query to highlight semantically similar documents on the 2D UMAP plot.")

query = st.text_input("Enter search query:")
top_k = st.slider("Number of matches to highlight", min_value=10, max_value=2500, value=100)

similarity_measure = st.radio(
    "Similarity measure",
    ["Cosine", "Euclidean", "Manhattan (L1)"],
    horizontal=True
)

# -------------------------------
# Search logic
# -------------------------------
if query:
    with st.spinner("Encoding and searching..."):
        query_embedding = model.encode(query, convert_to_tensor=True)
        query_numpy = query_embedding.cpu().numpy().reshape(1, -1)
        query_pca = pca_model.transform(query_numpy)
        query_umap = umap_model.transform(query_pca)

        if similarity_measure == "Cosine":
            scores = util.cos_sim(query_embedding, embeddings)[0]
        elif similarity_measure == "Euclidean":
            scores = -torch.cdist(query_embedding, embeddings, p=2)[0]
        elif similarity_measure == "Manhattan (L1)":
            scores = -torch.cdist(query_embedding, embeddings, p=1)[0]

        top_results = scores.argsort(descending=True)
        highlight_indices = top_results[:top_k].cpu().numpy()

    documents = docs.title_narrative
    labels = ["Match" if i in highlight_indices else "Other" for i in range(len(documents))]

    df = pd.DataFrame({
        "UMAP_1": umap_2d[:, 0],
        "UMAP_2": umap_2d[:, 1],
        "Label": labels,
        "Text": documents
    })

    df["Title"] = df["Text"].str.slice(0, 100) + "..."
    df["Index"] = df.index

    color_discrete_map = {"Match": "red", "Other": "lightgray"}

    fig = px.scatter(
        df,
        x="UMAP_1",
        y="UMAP_2",
        color="Label",
        color_discrete_map=color_discrete_map,
        hover_data={"Text": False, "Title": True, "Index": True, "UMAP_1": False, "UMAP_2": False},
        opacity=0.7,
        title=f"Top {top_k} semantic matches for: '{query}' ({similarity_measure})",
        width=900,
        height=700
    )

    fig.add_trace(go.Scatter(
        x=[query_umap[0][0]], y=[query_umap[0][1]],
        mode='markers+text',
        marker=dict(size=10, color='blue', symbol='x'),
        name='Query',
        text=['Query'], textposition='top center'
    ))

    fig.update_traces(marker=dict(size=4))
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Top 10 matched documents")
    for rank, idx in enumerate(highlight_indices[:10], start=1):
        st.markdown(f"{rank}. {documents.iloc[idx]}")
else:
    st.info("Enter a search query to begin.")
