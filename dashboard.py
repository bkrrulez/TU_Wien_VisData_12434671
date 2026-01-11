import streamlit as st
import pandas as pd
import plotly.express as px

# =====================================================
# Page configuration
# =====================================================
st.set_page_config(
    page_title="Political Freedom & Terrorism Dashboard",
    layout="wide"
)

# =====================================================
# Card-style CSS
# =====================================================
st.markdown(
    """
    <style>
    .card {
        border: 1px solid #dddddd;
        border-radius: 8px;
        padding: 16px;
        background-color: white;
        margin-bottom: 16px;
    }
    .card-title {
        font-weight: 600;
        font-size: 18px;
        margin-bottom: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# =====================================================
# Load data
# =====================================================
@st.cache_data
def load_data():
    return pd.read_excel("merged_freedom_gtd.xlsx")

df = load_data()

# =====================================================
# Recreate clustering dataframe
# =====================================================
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

cluster_df = df[[
    "Country/Territory",
    "year",
    "PR rating",
    "CL rating",
    "incidents"
]].dropna()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(
    cluster_df[["PR rating", "CL rating", "incidents"]]
)

kmeans = KMeans(n_clusters=5, random_state=42)
cluster_df["cluster"] = kmeans.fit_predict(X_scaled)

cluster_labels = {
    1: "Free & peaceful",
    0: "Moderately free & low violence",
    4: "Highly repressive but stable",
    2: "Repressive & high violence",
    3: "Extreme violence contexts"
}

cluster_df["cluster_label"] = cluster_df["cluster"].map(cluster_labels)

# =====================================================
# GLOBAL cluster order & color sync
# =====================================================
CLUSTER_ORDER = [
    "Free & peaceful",
    "Moderately free & low violence",
    "Highly repressive but stable",
    "Repressive & high violence",
    "Extreme violence contexts"
]

CLUSTER_COLORS = {
    "Free & peaceful": px.colors.qualitative.Safe[0],
    "Moderately free & low violence": px.colors.qualitative.Safe[1],
    "Highly repressive but stable": px.colors.qualitative.Safe[2],
    "Repressive & high violence": px.colors.qualitative.Safe[3],
    "Extreme violence contexts": px.colors.qualitative.Safe[4],
}

cluster_df["cluster_label"] = pd.Categorical(
    cluster_df["cluster_label"],
    categories=CLUSTER_ORDER,
    ordered=True
)

# =====================================================
# Title & description
# =====================================================
st.title("Political Freedom and Terrorism")
st.markdown(
    """
This interactive dashboard explores how **political freedom** and **terrorism**
relate across countries and over time. The visualizations summarize structural
patterns identified through clustering and allow interactive exploration.
"""
)

# =====================================================
# Sidebar filters
# =====================================================
st.sidebar.header("Filters")

year_range = st.sidebar.slider(
    "Select year range",
    int(cluster_df["year"].min()),
    int(cluster_df["year"].max()),
    (
        int(cluster_df["year"].min()),
        int(cluster_df["year"].max())
    )
)

filtered_df = cluster_df[
    (cluster_df["year"] >= year_range[0]) &
    (cluster_df["year"] <= year_range[1])
]

# =====================================================
# Dashboard layout
# =====================================================
row1_col1, row1_col2 = st.columns(2)
row2_col1, row2_col2 = st.columns(2)

# =====================================================
# Visualization 1 — Cluster overview
# =====================================================
with row1_col1:
    with st.container():
        st.markdown("""
            <div class="card">
                <div class="card-title">Country–Year Clusters</div>
        """, unsafe_allow_html=True)

        fig_v1 = px.scatter(
            filtered_df,
            x="PR rating",
            y="incidents",
            color="cluster_label",
            category_orders={"cluster_label": CLUSTER_ORDER},
            color_discrete_map=CLUSTER_COLORS,
            hover_data={
                "Country/Territory": True,
                "year": True,
                "incidents": True
            }
        )

        fig_v1.update_yaxes(type="log")
        fig_v1.update_traces(marker=dict(opacity=0.65))

        st.plotly_chart(fig_v1, use_container_width=True)

        st.markdown("</div>", unsafe_allow_html=True)

# =====================================================
# Visualization 2 — Cluster sizes
# =====================================================
with row1_col2:
    with st.container():
        st.markdown("""
            <div class="card">
                <div class="card-title">Cluster Sizes</div>
        """, unsafe_allow_html=True)

        cluster_counts = (
            filtered_df["cluster_label"]
            .value_counts()
            .reindex(CLUSTER_ORDER)
            .fillna(0)
            .rename_axis("Cluster")
            .reset_index(name="Count")
        )

        fig_v2 = px.bar(
            cluster_counts,
            x="Cluster",
            y="Count",
            color="Cluster",
            category_orders={"Cluster": CLUSTER_ORDER},
            color_discrete_map=CLUSTER_COLORS
        )

        st.plotly_chart(fig_v2, use_container_width=True)

        st.markdown("</div>", unsafe_allow_html=True)

# =====================================================
# Visualization 3 — Country time series
# =====================================================
with row2_col1:
    with st.container():
        st.markdown("""
            <div class="card">
                <div class="card-title">Terrorist Incidents Over Time</div>
        """, unsafe_allow_html=True)

        country = st.selectbox(
            "Select a country",
            sorted(filtered_df["Country/Territory"].unique())
        )

        country_df = filtered_df[
            filtered_df["Country/Territory"] == country
        ]

        fig_v3 = px.line(
            country_df,
            x="year",
            y="incidents",
            markers=True,
            hover_data=["cluster_label"]
        )

        fig_v3.update_yaxes(type="log")

        st.plotly_chart(fig_v3, use_container_width=True)

        st.markdown("</div>", unsafe_allow_html=True)

# =====================================================
# Visualization 4 — Distribution per cluster
# =====================================================
with row2_col2:
    with st.container():
        st.markdown("""
            <div class="card">
                <div class="card-title">Distribution of Terrorist Incidents</div>
        """, unsafe_allow_html=True)

        fig_v4 = px.box(
            filtered_df,
            x="cluster_label",
            y="incidents",
            color="cluster_label",
            category_orders={"cluster_label": CLUSTER_ORDER},
            color_discrete_map=CLUSTER_COLORS
        )

        fig_v4.update_yaxes(type="log")

        st.plotly_chart(fig_v4, use_container_width=True)

        st.markdown("</div>", unsafe_allow_html=True)

# =====================================================
# Footer
# =====================================================
st.markdown(
    """
**Notes:**  
• Clusters represent **country–year profiles**, not countries.  
• Log scales are used to handle extreme values in terrorism data.  
• The dashboard is exploratory and descriptive, not predictive.
"""
)
