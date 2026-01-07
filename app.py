import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="ICU Patient Anomaly Detection",
    layout="centered"
)

st.title("🏥 ICU Patient Anomaly Detection using DBSCAN")
st.write("This app detects abnormal ICU patient patterns based on hospital interactions.")

# ---------------- FILE UPLOAD ----------------
uploaded_file = st.file_uploader("Upload ICU Patient Dataset (CSV)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("📄 Dataset Preview")
    st.dataframe(df.head())

    # ---------------- DROP COLUMNS ----------------
    drop_cols = [
        'hadm_id',
        'gender', 'admit_type', 'admit_location',
        'AdmitDiagnosis', 'insurance', 'religion',
        'marital_status', 'ethnicity', 'AdmitProcedure',
        'LOSgroupNum', 'ExpiredHospital'
    ]

    df_clean = df.drop(columns=drop_cols, errors='ignore')

    st.subheader("🧹 Cleaned Dataset (Numeric Features)")
    st.write(df_clean.columns.tolist())

    # ---------------- SCALING ----------------
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_clean)

    # ---------------- EPS SELECTION ----------------
    st.subheader("📈 K-Distance Graph (for eps selection)")

    neighbors = NearestNeighbors(n_neighbors=5)
    neighbors_fit = neighbors.fit(X_scaled)
    distances, indices = neighbors_fit.kneighbors(X_scaled)

    distances = np.sort(distances[:, 4])

    fig, ax = plt.subplots()
    ax.plot(distances)
    ax.set_title("K-distance Graph")
    ax.set_xlabel("Data Points")
    ax.set_ylabel("Distance")
    st.pyplot(fig)

    st.info("👆 Choose eps value from the elbow point")

    # ---------------- DBSCAN INPUT ----------------
    eps = st.slider("Select eps value", 0.1, 5.0, 1.2, 0.1)
    min_samples = st.slider("Select min_samples", 3, 10, 5)

    # ---------------- APPLY DBSCAN ----------------
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    clusters = dbscan.fit_predict(X_scaled)

    df_clean['Cluster'] = clusters

    # ---------------- RESULTS ----------------
    st.subheader("📊 Cluster Distribution")
    st.write(df_clean['Cluster'].value_counts())

    # ---------------- VISUALIZATION ----------------
    st.subheader("📉 DBSCAN Clustering Visualization")

    fig2, ax2 = plt.subplots()
    sns.scatterplot(
        x=df_clean['LOSdays'],
        y=df_clean['TotalNumInteract'],
        hue=df_clean['Cluster'],
        palette='Set1',
        ax=ax2
    )
    ax2.set_title("ICU Patient Clustering (DBSCAN)")
    ax2.set_xlabel("Length of Stay (days)")
    ax2.set_ylabel("Total Hospital Interactions")

    st.pyplot(fig2)

    # ---------------- INTERPRETATION ----------------
    st.subheader("🧠 Interpretation")
    st.markdown("""
    - **Cluster -1** represents **abnormal / high-risk patients**
    - Other clusters represent **normal ICU patient behavior**
    - DBSCAN automatically detects anomalies without predefined labels
    """)

else:
    st.warning("👆 Please upload a CSV file to continue")
