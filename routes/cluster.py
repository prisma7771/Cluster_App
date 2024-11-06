import joblib
import streamlit as st
import pandas as pd
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import plotly.graph_objects as go
from utils.computation import compute_tsne
from utils.utils import pre_process


kmeans_model = joblib.load("data/kmeans_model_pca_firefly.joblib")

old_data = pd.read_csv("data/data_cluster_ori.csv")

old_data.rename(columns={"predicted_cluster": "Cluster"}, inplace=True)
old_data["DataType"] = "Original"


def calculate_data(selected_items):
    result = {
        "jml_pembelian": 0,
        "total_harga": 0,
        "kategori_Coffee": 0,
        "kategori_Food": 0,
        "kategori_Non-Coffee": 0,
    }

    for item, quantity in selected_items.items():
        price_info = ITEM_PRICES[item]
        category = price_info["category"]
        total_price = price_info["price"] * quantity

        result["jml_pembelian"] += quantity
        result["total_harga"] += total_price
        result[f"kategori_{category}"] += total_price

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
    X_tsne = compute_tsne(combined_data, 3)

    set2_colors = cm.Set2.colors
    color_map = {i: mcolors.to_hex(set2_colors[i]) for i in range(len(set2_colors))}

    tsne_df = pd.DataFrame(
        data=X_tsne, columns=["component_1", "component_2", "component_3"]
    )
    tsne_df["Cluster"] = pred_cluster
    tsne_df["DataType"] = data_type

    fig = go.Figure()

    for cluster in set(pred_cluster):
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
                ),
                name=f"Cluster {cluster} - New",
                showlegend=True,
            )
        )

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis=dict(title="Component 1"),
            yaxis=dict(title="Component 2"),
            zaxis=dict(title="Component 3"),
        ),
        legend=dict(title="Clusters and Data Type", x=0.9, y=0.9),
    )

    st.plotly_chart(fig)


st.sidebar.title("Clustering Menu")
menu_option = st.sidebar.selectbox(
    "Select an option",
    ["Upload Data", "Input New Data"],
)


if menu_option == "Upload Data":
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
    if uploaded_file is not None:
        data_uploaded = pd.read_csv(uploaded_file)
        data_processed = pre_process(data_uploaded)

        col1, col2 = st.columns(2)

        with col1:
            st.write("Data Preview:")
            st.dataframe(data_uploaded)

        with col2:
            st.write("Processed Data Preview:")
            st.dataframe(data_processed)

        if st.button("Predict Clusters"):
            predictions = kmeans_model.predict(data_processed)

            data_processed["Cluster"] = predictions
            st.write("Predicted Clusters:")
            st.dataframe(data_processed)

            new_data = data_processed.copy()
            new_data["DataType"] = "New"

            combined_data = pd.concat([old_data, new_data], ignore_index=True)

            features_only = combined_data.drop(columns=["Cluster", "DataType"])

            plot_3d_tsne_new(
                features_only, combined_data["Cluster"], combined_data["DataType"]
            )


elif menu_option == "Input New Data":
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

    CATEGORY_ITEMS = {
        category: [
            item for item, info in ITEM_PRICES.items() if info["category"] == category
        ]
        for category in CATEGORIES
    }

    if "new_data" not in st.session_state:
        st.session_state["new_data"] = []

    st.title("Input New Order Data")

    selected_items = {}

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

            st.markdown("<div class='v-center'>", unsafe_allow_html=True)
            if st.button(f"Add {item} to data", key=f"add_{category}"):
                if quantity > 0:
                    price = ITEM_PRICES[item]["price"]
                    item_found = False
                    for existing_item in st.session_state["new_data"]:
                        if existing_item["item"] == item:
                            existing_item["quantity"] += quantity
                            item_found = True
                            break

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

    if st.session_state["new_data"]:
        st.write("New Data:")
        new_data_df = pd.DataFrame(st.session_state["new_data"])
        st.dataframe(new_data_df)

        jml_pembelian = new_data_df["quantity"].sum()

        total_harga = (new_data_df["quantity"] * new_data_df["price"]).sum()

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

        average_price = total_harga / jml_pembelian

        ratio_Coffee = kategori_Coffee / total_harga if total_harga != 0 else 0
        ratio_Food = kategori_Food / total_harga if total_harga != 0 else 0
        ratio_Non_Coffee = kategori_Non_Coffee / total_harga if total_harga != 0 else 0

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
        predictions = kmeans_model.predict(data_processed)

        data_processed["Cluster"] = predictions
        st.write("Predicted Clusters:")
        st.dataframe(data_processed)

        new_data = data_processed.copy()
        new_data["DataType"] = "New"

        combined_data = pd.concat([old_data, new_data], ignore_index=True)

        features_only = combined_data.drop(columns=["Cluster", "DataType"])

        plot_3d_tsne_new(
            features_only, combined_data["Cluster"], combined_data["DataType"]
        )
