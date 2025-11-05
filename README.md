# 💳 Credit Card Transaction Analysis – Graph Clustering for Fraud Detection

This project implements **graph-based unsupervised clustering** (using **Spectral Clustering**) to detect fraudulent credit card transactions.  
Instead of analyzing each transaction independently, it models relationships between transactions as a **graph network**, allowing the detection of **fraud rings** and coordinated fraudulent behavior.

---

## 🎯 Project Objective

- To analyze credit card transaction data and identify **fraudulent patterns**.
- To model transaction relationships as a **graph** based on similarity.
- To apply **Spectral Clustering** for grouping similar transactions.
- To identify the cluster with the **highest fraud ratio** (potential fraud ring).
- To visualize results using **PCA** and **graph network plots** in a **Streamlit web app**.

---

## ⚙️ Technologies & Tools Used

| Category | Tools / Libraries |
|-----------|------------------|
| **Programming Language** | Python |
| **Framework** | Streamlit |
| **Libraries** | pandas, numpy, networkx, scikit-learn, matplotlib, seaborn |
| **Algorithm** | Spectral Clustering |
| **Visualization** | PCA (Principal Component Analysis), Network Graphs |
| **Dataset Source** | [Kaggle – Credit Card Fraud Detection Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) |

---

## 🧩 Workflow

1. **Data Preprocessing** – Scale the `Amount` feature and balance legitimate vs. fraudulent transactions.  
2. **Graph Construction** –  
   - Compute pairwise Euclidean distances.  
   - Convert to similarity matrix using Gaussian kernel.  
   - Threshold to form adjacency matrix (edges between similar transactions).  
3. **Spectral Clustering** – Cluster transactions based on graph structure.  
4. **Fraud Ring Detection** – Identify the cluster with the **highest fraud ratio**.  
5. **Visualization** – Use PCA plots and graph diagrams to visualize fraud separation.

---

## 📊 Example Output Metrics

| Metric | Example Value |
|--------|----------------|
| Total Transactions (Sample) | 5000 |
| Actual Frauds | 492 |
| Detected Frauds | 460 |
| False Positives | 32 |
| F1-Score | 0.93 |

---

## 🖥️ Streamlit Web App Features

- Upload any CSV dataset with similar structure.
- Adjust parameters:  
  - Number of clusters (k)  
  - Similarity threshold  
  - Legit-to-fraud sampling ratio  
- View fraud distribution by cluster.
- Visualize results interactively with PCA plots and network graphs.

---

## 🧠 How to Run Locally

1. **Clone the repository**
   ```bash
   git clone https://github.com/KrrishTayal/Credit-Card-Transaction-Analysis-using-Graph-clustering.git
   cd Credit-Card-Transaction-Analysis-using-Graph-clustering

**2.Install dependencies**

   
   pip install -r requirements.txt


**Run the Streamlit app**

   
   streamlit run app.py


**Upload the dataset
Download the CSV from Kaggle:**
👉 https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

**🌐 Deployment**

This project can be deployed for free on Streamlit Cloud:

Push the repo to GitHub.

Go to https://share.streamlit.io
.

Select your repo and main file (app.py).

Deploy your app and share the link.

**🧾 Author

Krrish Tayal**
