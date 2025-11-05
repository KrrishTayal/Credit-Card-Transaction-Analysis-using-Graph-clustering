import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.cluster import SpectralClustering
from sklearn.metrics import classification_report, f1_score
from sklearn.decomposition import PCA


# Streamlit Configuration

st.set_page_config(page_title="Credit Card Fraud Detection - Graph Clustering", layout="wide")
st.title("Credit Card Transaction Analysis | Graph Clustering for Fraud Detection")


# Sidebar - Parameters

st.sidebar.header(" Graph Clustering Parameters")
k_optimal = st.sidebar.slider("Number of Clusters (k)", 2, 10, 3)
threshold = st.sidebar.slider("Similarity Threshold", 0.5, 0.99, 0.9, 0.01)
sample_ratio = st.sidebar.slider("Legit : Fraud Ratio", 2, 20, 10, 1)
show_graph = st.sidebar.checkbox("Show Graph Network (for small samples)", False)


# Data Upload

uploaded_file = st.file_uploader(" Upload Credit Card Transactions CSV", type=["csv"])
if uploaded_file:
    data = pd.read_csv(uploaded_file)
else:
    data = pd.read_csv("creditcard.csv")

st.markdown(f"**Total Transactions:** {len(data)} | **Fraudulent:** {data['Class'].sum()}")


# Data Preparation

fraud_data = data[data['Class'] == 1].copy()
legit_sample = data[data['Class'] == 0].sample(n=len(fraud_data)*sample_ratio, random_state=42).copy()
graph_data = pd.concat([fraud_data, legit_sample]).reset_index(drop=True)

X_graph = graph_data.drop(['Time', 'Class'], axis=1)
y_graph = graph_data['Class']

# Standardize amount feature
scaler = StandardScaler()
X_graph['Amount'] = scaler.fit_transform(X_graph['Amount'].values.reshape(-1, 1))

st.info(f" Sample size used for graph construction: {len(graph_data)} transactions")


# Graph Construction

st.subheader(" Step 1: Graph Construction from Similarities")

dist_matrix = euclidean_distances(X_graph)
sigma = np.std(dist_matrix)
similarity_matrix = np.exp(-dist_matrix**2 / (2. * sigma**2))
adjacency_matrix = np.where(similarity_matrix > threshold, similarity_matrix, 0)

G = nx.from_numpy_array(adjacency_matrix)
col1, col2 = st.columns(2)
col1.metric("Graph Nodes", G.number_of_nodes())
col2.metric("Graph Edges", G.number_of_edges())


# Spectral Clustering


# Spectral Clustering Results

st.subheader(" Step 2: Spectral Clustering on Graph")

sc = SpectralClustering(n_clusters=k_optimal, affinity='precomputed', random_state=42)
graph_data['Cluster'] = sc.fit_predict(adjacency_matrix)

cluster_counts = graph_data.groupby('Cluster')['Class'].value_counts().unstack(fill_value=0)
cluster_counts['Fraud_Ratio'] = cluster_counts[1] / (cluster_counts[0] + cluster_counts[1])
fraud_cluster_label = cluster_counts['Fraud_Ratio'].idxmax()

graph_data['Prediction'] = graph_data['Cluster'].apply(lambda x: 1 if x == fraud_cluster_label else 0)

#  Count of actual vs detected frauds
total_frauds = int(y_graph.sum())
detected_frauds = int(graph_data.loc[graph_data['Prediction'] == 1, 'Class'].sum())
false_positives = int(graph_data.loc[(graph_data['Prediction'] == 1) & (graph_data['Class'] == 0)].shape[0])

sc_f1 = f1_score(y_graph, graph_data['Prediction'])

# Display key metrics
st.write("#  Cluster Summary")
st.dataframe(cluster_counts.style.background_gradient(cmap="YlOrRd"))
st.success(f" Cluster with highest fraud ratio (Fraud Ring): {fraud_cluster_label}")

#  Highlight fraud detection numbers
col1, col2, col3 = st.columns(3)
col1.metric("Total Fraud Transactions (Actual)", total_frauds)
col2.metric("Frauds Detected by Clustering", detected_frauds)
col3.metric("False Positives", false_positives)

st.metric("Spectral Clustering F1-Score", f"{sc_f1:.4f}")



# PCA Visualization

st.subheader(" Step 3: Visualization of Clusters (PCA Projection)")

pca = PCA(n_components=2)
principal_components = pca.fit_transform(X_graph)
pca_df = pd.DataFrame(principal_components, columns=['PC1', 'PC2'])
pca_df['Cluster'] = graph_data['Cluster']
pca_df['Actual_Class'] = graph_data['Class']

fig, ax = plt.subplots(figsize=(10, 6))
sns.scatterplot(
    x='PC1', y='PC2',
    hue='Cluster', style='Actual_Class',
    palette='Spectral', data=pca_df, alpha=0.7, s=60
)
plt.title(f"Spectral Clustering Results (k={k_optimal})")
st.pyplot(fig)


# Graph Visualization 

if show_graph:
    st.subheader(" Step 4: Graph Network Visualization")
    fig, ax = plt.subplots(figsize=(8, 6))
    pos = nx.spring_layout(G, seed=42)
    nx.draw(G, pos, node_size=20, node_color='skyblue', alpha=0.6, with_labels=False)
    st.pyplot(fig)


# Final Summary

st.markdown("---")
st.subheader(" Final Insights")
st.markdown(f"""
- The fraud detection system successfully modeled transactions as a **graph network**.  
- Using **Spectral Clustering**, transactions were grouped based on relational similarity.  
- The cluster with the **highest fraud ratio** represents the likely **fraud ring**.  
- The achieved **F1-score ({sc_f1:.4f})** shows the system's accuracy in isolating fraudulent transactions.
""")
