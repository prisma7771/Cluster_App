import streamlit as st
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


@st.cache_data
def pipeline_ori(_df):
    preprocessor = Pipeline(
        [
            (
                "scaler",
                StandardScaler(),
            ),
        ]
    )

    clusterer = Pipeline(
        [
            (
                "kmeans",
                KMeans(
                    n_clusters=5,
                    init="k-means++",
                    n_init=10,
                    max_iter=1000,
                    random_state=42,
                ),
            ),
        ]
    )

    pipe = Pipeline([("preprocessor", preprocessor), ("clusterer", clusterer)])

    pipe.fit(_df)

    return pipe


@st.cache_data
def pipeline_pca_firefly(_df):
    preprocessor = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=4)),
        ]
    )

    best_centroids = [
        [-1.12126981, -2.25502389, 0.729677, 2.42538169],
        [-0.38103954, -2.6732293, -2.03745063, -1.88134444],
        [-1.13828433, 2.87735497, -0.55055388, -1.88134444],
        [0.41608554, -3.33765317, -2.03745063, -0.86960191],
        [-0.92351411, -0.06942669, 1.29904736, 1.29209624],
    ]

    clusterer = Pipeline(
        [
            (
                "kmeans",
                KMeans(
                    n_clusters=5,
                    init=best_centroids,
                    n_init=1,
                    max_iter=1000,
                    random_state=42,
                ),
            ),
        ]
    )

    pipe = Pipeline([("preprocessor", preprocessor), ("clusterer", clusterer)])

    pipe.fit(_df)

    return pipe
