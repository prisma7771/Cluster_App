import streamlit as st
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


@st.cache_data
def pipeline_ori(_df):
    # Step 1: Create PreProcessor Pipeline
    preprocessor = Pipeline(
        [
            (
                "scaler",
                StandardScaler(),
            ),  # Standardize features by removing the mean and scaling to unit variance
        ]
    )

    # Step 2: Create Clusterer Pipeline
    clusterer = Pipeline(
        [
            (
                "kmeans",
                KMeans(
                    n_clusters=5,  # Define the number of clusters
                    init="k-means++",  # Smart initialization of centroids
                    n_init=10,  # Number of times the k-means algorithm will run with different centroid seeds
                    max_iter=1000,  # Maximum number of iterations for a single run
                    random_state=42,  # Ensure reproducibility
                ),
            ),
        ]
    )

    # Step 3: Combine PreProcessor and Clusterer into a Full Pipeline
    pipe = Pipeline([("preprocessor", preprocessor), ("clusterer", clusterer)])

    # Fit the pipeline on the provided data
    pipe.fit(_df)

    return pipe


@st.cache_data
def pipeline_pca_firefly(_df):
    # Create Pipeline PreProcessor
    preprocessor = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=4)),
        ]
    )

    # Firefly Run
    best_centroids = [
        [-1.12126981, -2.25502389, 0.729677, 2.42538169],
        [-0.38103954, -2.6732293, -2.03745063, -1.88134444],
        [-1.13828433, 2.87735497, -0.55055388, -1.88134444],
        [0.41608554, -3.33765317, -2.03745063, -0.86960191],
        [-0.92351411, -0.06942669, 1.29904736, 1.29209624],
    ]

    # Step 3: Define the clusterer pipeline with the best centroids
    clusterer = Pipeline(
        [
            (
                "kmeans",
                KMeans(
                    n_clusters=5,
                    init=best_centroids,
                    #                init="k-means++",
                    n_init=1,  # We only need one initialization since we already have the centroids
                    max_iter=1000,
                    random_state=42,
                ),
            ),
        ]
    )

    # Step 4: Combine the preprocessor and clusterer into a single pipeline
    pipe = Pipeline([("preprocessor", preprocessor), ("clusterer", clusterer)])

    # Step 5: Fit the complete pipeline to your data
    pipe.fit(_df)

    return pipe
