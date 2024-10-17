import joblib
import streamlit as st
import pandas as pd
import matplotlib.cm as cm
import matplotlib.colors as mcolors

import plotly.graph_objects as go

from utils.computation import compute_tsne3
from utils.utils import pre_process

# Load your model pipeline (ensure the correct path to your .joblib file)
kmeans_model = joblib.load("data/kmeans_model_pca_firefly.joblib")

# Load precomputed t-SNE results for old data
old_data = pd.read_csv("data/data_cluster_ori.csv")

# Rename cluster column in the old data for consistency
old_data.rename(columns={"predicted_cluster": "Cluster"}, inplace=True)
old_data["DataType"] = "Original"


# Function to calculate the data
def calculate_data(selected_items):
    # Initialize the results dictionary
    result = {
        "jml_pembelian": 0,
        "total_harga": 0,
        "kategori_Coffee": 0,
        "kategori_Food": 0,
        "kategori_Non-Coffee": 0,
    }

    # Calculate totals and categories
    for item, quantity in selected_items.items():
        price_info = ITEM_PRICES[item]
        category = price_info["category"]
        total_price = price_info["price"] * quantity

        result["jml_pembelian"] += quantity
        result["total_harga"] += total_price
        result[f"kategori_{category}"] += total_price

    # Calculate average price and ratios
    result["average_price"] = (
        result["total_harga"] / result["jml_pembelian"]
        if result["jml_pembelian"] > 0
        else 0
    )
    for category in CATEGORIES:
        result[f"ratio_{category}"] = (
            result[f"kategori_{category}"] / result["total_harga"]
            if result["total_harga"] > 0
            else 0
        )

    return result


def plot_3d_tsne_new(
    combined_data, pred_cluster, data_type, title="t-SNE 3D Visualization"
):
    # Get the cached 3D t-SNE result
    X_tsne = compute_tsne3(combined_data)

    # Define a colormap similar to Matplotlib's Set2
    set2_colors = cm.Set2.colors
    color_map = {i: mcolors.to_hex(set2_colors[i]) for i in range(len(set2_colors))}

    # Create a DataFrame with t-SNE results and cluster labels
    tsne_df = pd.DataFrame(
        data=X_tsne, columns=["component_1", "component_2", "component_3"]
    )
    tsne_df["Cluster"] = pred_cluster
    tsne_df["DataType"] = data_type

    # Initialize a 3D scatter plot figure
    fig = go.Figure()

    # Adding scatter plot traces for each cluster in both original and new data
    for cluster in set(pred_cluster):  # Iterate through unique cluster labels
        # Plot Original Data for the Cluster
        original_cluster_data = tsne_df[
            (tsne_df["Cluster"] == cluster) & (tsne_df["DataType"] == "Original")
        ]
        fig.add_trace(
            go.Scatter3d(
                x=original_cluster_data["component_1"],
                y=original_cluster_data["component_2"],
                z=original_cluster_data["component_3"],
                mode="markers",
                marker=dict(size=5, color=color_map[cluster], symbol="circle"),
                name=f"Cluster {cluster} - Original",
                showlegend=True,
            )
        )

        # Plot New Data for the Cluster with a different marker style
        new_cluster_data = tsne_df[
            (tsne_df["Cluster"] == cluster) & (tsne_df["DataType"] == "New")
        ]
        fig.add_trace(
            go.Scatter3d(
                x=new_cluster_data["component_1"],
                y=new_cluster_data["component_2"],
                z=new_cluster_data["component_3"],
                mode="markers",
                marker=dict(
                    size=7,
                    color=color_map[cluster],
                    symbol="x",
                    line=dict(width=1, color="black"),
                ),  # Add outline
                name=f"Cluster {cluster} - New",
                showlegend=True,
            )
        )

    # Update layout for 3D plot
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis=dict(title="Component 1"),
            yaxis=dict(title="Component 2"),
            zaxis=dict(title="Component 3"),
        ),
        legend=dict(title="Clusters and Data Type", x=0.9, y=0.9),
    )

    # Use Streamlit to display the plot
    st.plotly_chart(fig)


# Sidebar for menu options
st.sidebar.title("Clustering Menu")
menu_option = st.sidebar.selectbox(
    "Select an option",
    ["Upload Data", "Input New Data"],
)

# Upload data
if menu_option == "Upload Data":
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        data_processed = pre_process(data)

        col1, col2 = st.columns(2)

        with col1:
            st.write("Data Preview:")
            st.dataframe(data)

        with col2:
            st.write("Processed Data Preview:")
            st.dataframe(data_processed)

        if st.button("Predict Clusters"):
            # Predict using the combined pipeline
            predictions = kmeans_model.predict(data_processed)

            # Add predictions to the processed data for better visualization
            data_processed["Cluster"] = predictions
            st.write("Predicted Clusters:")
            st.dataframe(data_processed)

            # Load new data and preprocess it
            new_data = (
                data_processed.copy()
            )  # Assuming 'data_processed' is already preprocessed
            new_data["DataType"] = "New"

            # Combine the old and new data
            combined_data = pd.concat([old_data, new_data], ignore_index=True)

            # Select only the feature columns (exclude 'Cluster' and 'DataType' for t-SNE)
            features_only = combined_data.drop(columns=["Cluster", "DataType"])

            # plot
            plot_3d_tsne_new(
                features_only, combined_data["Cluster"], combined_data["DataType"]
            )


elif menu_option == "Input New Data":
    # Step 1: Define the price and category information
    ITEM_PRICES = {
        "Cappucinno": {"price": 18000, "category": "Coffee"},
        "Mix Platter": {"price": 15000, "category": "Food"},
        "Tea": {"price": 5000, "category": "Non-Coffee"},
        "Americano": {"price": 12000, "category": "Coffee"},
        "Pisang Goreng": {"price": 9000, "category": "Food"},
        "Mojito": {"price": 15000, "category": "Non-Coffee"},
        "Vietnam Drip": {"price": 12000, "category": "Coffee"},
        "Espresso": {"price": 10000, "category": "Coffee"},
        "Latte": {"price": 18000, "category": "Coffee"},
        "Strawberry Squash": {"price": 15000, "category": "Non-Coffee"},
        "Mendoan": {"price": 8000, "category": "Food"},
        "Milk Shake": {"price": 15000, "category": "Non-Coffee"},
        "Tahu Bakso": {"price": 12000, "category": "Food"},
        "Kopi Susu": {"price": 18000, "category": "Coffee"},
        "Kopi Aren": {"price": 18000, "category": "Coffee"},
        "Dalgona Coffee": {"price": 18000, "category": "Coffee"},
        "Waffle": {"price": 14000, "category": "Food"},
        "Roti bakar": {"price": 8000, "category": "Food"},
        "Strawberry Tea": {"price": 7000, "category": "Non-Coffee"},
        "Fruit Latte": {"price": 18000, "category": "Coffee"},
        "Teh Tarik": {"price": 8000, "category": "Non-Coffee"},
        "Espresso Boom": {"price": 16000, "category": "Coffee"},
        "Vanilla Latte": {"price": 18000, "category": "Coffee"},
        "Jahe Susu": {"price": 8000, "category": "Non-Coffee"},
        "Mochacino": {"price": 18000, "category": "Coffee"},
    }

    CATEGORIES = ["Coffee", "Food", "Non-Coffee"]

    # Filter items by category
    CATEGORY_ITEMS = {
        category: [
            item for item, info in ITEM_PRICES.items() if info["category"] == category
        ]
        for category in CATEGORIES
    }

    # Initialize session state to store new data if it does not already exist
    if "new_data" not in st.session_state:
        st.session_state["new_data"] = []

    # Streamlit app
    st.title("Input New Order Data")

    # Create select boxes and input fields for each category
    selected_items = {}

    # For each category, provide a dropdown menu and input for quantity
    for category in CATEGORIES:
        col1, col2, col3 = st.columns([5, 3, 2])

        with col1:
            item = st.selectbox(
                f"Select {category} item", CATEGORY_ITEMS[category], key=category
            )
            quantity = st.number_input(
                f"Enter quantity for {item}:",
                min_value=1,
                step=1,
                key=f"qty_{category}",
            )

        with col2:
            # Center-align the button vertically
            st.markdown(
                """
            <style>
            .v-center {
                display: flex;
                align-items: center;
                justify-content: center;
                flex-direction: column;
                padding: 20px; /* Adjust padding as needed */
            }
            </style>
            """,
                unsafe_allow_html=True,
            )

            # Creating a div with a class to center the button
            st.markdown("<div class='v-center'>", unsafe_allow_html=True)
            if st.button(f"Add {item} to data", key=f"add_{category}"):
                if quantity > 0:
                    price = ITEM_PRICES[item]["price"]
                    item_found = False
                    for existing_item in st.session_state["new_data"]:
                        if existing_item["item"] == item:
                            existing_item["quantity"] += quantity  # Update the quantity
                            item_found = True
                            break

                    # If the item was not found, append it as a new entry
                    if not item_found:
                        st.session_state["new_data"].append(
                            {
                                "item": item,
                                "category": category,
                                "quantity": quantity,
                                "price": price,
                            }
                        )

                    with col3:
                        st.markdown("<div class='v-center'>", unsafe_allow_html=True)
                        st.success(f"Add {item} {quantity}x to DataFrame!")
                else:
                    with col3:
                        st.markdown("<div class='v-center'>", unsafe_allow_html=True)
                        st.warning("Please add valid quantity!")

    if st.button("Clear DataFrame"):
        st.session_state["new_data"] = []

    # Display the accumulated data
    if st.session_state["new_data"]:
        st.write("New Data:")
        new_data_df = pd.DataFrame(st.session_state["new_data"])
        st.dataframe(new_data_df)

        # Total quantity (jml_pembelian)
        jml_pembelian = new_data_df["quantity"].sum()

        # Total price (total_harga)
        total_harga = (new_data_df["quantity"] * new_data_df["price"]).sum()

        # Total price per category
        kategori_Coffee = (
            new_data_df[new_data_df["category"] == "Coffee"]["quantity"]
            * new_data_df[new_data_df["category"] == "Coffee"]["price"]
        ).sum()
        kategori_Food = (
            new_data_df[new_data_df["category"] == "Food"]["quantity"]
            * new_data_df[new_data_df["category"] == "Food"]["price"]
        ).sum()
        kategori_Non_Coffee = (
            new_data_df[new_data_df["category"] == "Non-Coffee"]["quantity"]
            * new_data_df[new_data_df["category"] == "Non-Coffee"]["price"]
        ).sum()

        # Average price (average_price)
        average_price = total_harga / jml_pembelian

        # Ratio of each category (ratio_Coffee, ratio_Food, ratio_Non_Coffee)
        ratio_Coffee = kategori_Coffee / total_harga if total_harga != 0 else 0
        ratio_Food = kategori_Food / total_harga if total_harga != 0 else 0
        ratio_Non_Coffee = kategori_Non_Coffee / total_harga if total_harga != 0 else 0

        # Create the final summarized DataFrame
        data_processed = pd.DataFrame(
            {
                "jml_pembelian": [jml_pembelian],
                "total_harga": [total_harga],
                "kategori_Coffee": [kategori_Coffee],
                "kategori_Food": [kategori_Food],
                "kategori_Non_Coffee": [kategori_Non_Coffee],
                "average_price": [average_price],
                "ratio_Coffee": [ratio_Coffee],
                "ratio_Food": [ratio_Food],
                "ratio_Non_Coffee": [ratio_Non_Coffee],
            }
        )

        st.dataframe(data_processed)

    if st.button("Predict Clusters"):
        # Predict using the combined pipeline
        predictions = kmeans_model.predict(data_processed)

        # Add predictions to the processed data for better visualization
        data_processed["Cluster"] = predictions
        st.write("Predicted Clusters:")
        st.dataframe(data_processed)

        # Load new data and preprocess it
        new_data = data_processed.copy()
        new_data["DataType"] = "New"

        # Combine the old and new data
        combined_data = pd.concat([old_data, new_data], ignore_index=True)

        # Select only the feature columns (exclude 'Cluster' and 'DataType' for t-SNE)
        features_only = combined_data.drop(columns=["Cluster", "DataType"])

        # plot
        plot_3d_tsne_new(
            features_only, combined_data["Cluster"], combined_data["DataType"]
        )
