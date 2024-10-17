import streamlit as st
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def initialize_fireflies(n, k, X):
    """Initialize fireflies with random centroids"""
    n_samples, n_features = X.shape
    fireflies = []
    for _ in range(n):
        centroids = X[np.random.choice(n_samples, k, replace=False)]
        fireflies.append(centroids)
    return np.array(fireflies)

def fitness_function(centroids, X):
    """Evaluate the quality of the centroids using both SSE and silhouette score"""
    centroids = np.array(centroids).reshape(-1, X.shape[1])
    kmeans = KMeans(n_clusters=len(centroids), init=centroids, n_init=1, random_state=42).fit(X)
    labels = kmeans.labels_
    
    sse = kmeans.inertia_  # Sum of Squared Errors (SSE)
    
    if len(set(labels)) > 1:
        silhouette = silhouette_score(X, labels)
    else:
        silhouette = -1  # Poor silhouette score if only one cluster is found
    
    return sse, silhouette

def update_fireflies(fireflies, attractiveness, X, alpha=0.6, beta=1.0, gamma=1.0):
    n = len(fireflies)
    new_fireflies = np.copy(fireflies)
    for i in range(n):
        for j in range(n):
            if attractiveness[j][1] > attractiveness[i][1]:  # Compare based on silhouette score
                move = beta * np.exp(-gamma * np.linalg.norm(fireflies[i] - fireflies[j])**2) * (fireflies[j] - fireflies[i])
                noise = alpha * np.random.randn(*fireflies[i].shape)
                new_fireflies[i] = np.clip(fireflies[i] + move + noise, X.min(axis=0), X.max(axis=0))
    return new_fireflies

@st.cache_data
def firefly_algorithm(X, k=3, n_fireflies=10, n_iterations=50, alpha=0.6, beta=1.0, gamma=1.0, sse_weight=0.3, silhouette_weight=0.7):
    fireflies = initialize_fireflies(n_fireflies, k, X)
    best_centroids = None
    best_score = -float('inf')
    
    iteration_scores = []
    
    for iteration in range(n_iterations):
        # Evaluate fitness
        attractiveness = np.array([fitness_function(centroids, X) for centroids in fireflies])
        
        # Fitness function returns (SSE, silhouette_score)
        sse_scores = attractiveness[:, 0]
        silhouette_scores = attractiveness[:, 1]
        
        # Calculate composite scores based on weighted sum
        composite_scores = silhouette_weight * silhouette_scores - sse_weight * sse_scores
        
        current_best_score = np.max(composite_scores)  # Choose based on composite score
        
        # Update best solution
        if current_best_score > best_score:
            best_score = current_best_score
            best_centroids = fireflies[np.argmax(composite_scores)]
        
        # Record iteration data
        iteration_scores.append((iteration, best_score))
        
        # Update fireflies
        fireflies = update_fireflies(fireflies, attractiveness, X, alpha, beta, gamma)
    
    return best_centroids, best_score, iteration_scores
