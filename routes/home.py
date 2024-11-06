import io
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from utils.utils import load_data


def show_introduction():
    st.title("Homepage")
    st.markdown("""
    **Welcome to My K-Means App Optimized With Firefly And PCA!**

    Halo! Nama saya **Prisma Putra** dengan NIM **123190048**.
    Ini adalah proyek akhir saya untuk studi saya yang dibuat menggunakan *Streamlit* untuk pengembangan website dan *Python* sebagai bahasa utamanya.

    Untuk proyek ini, saya menggunakan library berikut:
    - **Pandas**: Untuk manipulasi dan analisis data
    - **NumPy**: Untuk operasi numerik
    - **Scikit-learn (sklearn)**: Untuk menerapkan algoritma machine learning
    - **Matplotlib**: Untuk generate plot dan render
    - **Seaborn**: Untuk high level customize plot
    - **Plotly**: Untuk generate interaktif plot 3d 
    - **Joblib**: Untuk menyimpan model hasil training  
    - **Jupyter Notebook**: Untuk pengujian dan pengembangan prototipe
    - **Streamlit**: Untuk membangun aplikasi web

    Untuk optimisasi, saya menggunakan:
    - **PCA (Principal Component Analysis)**: Untuk reduksi dimensi
    - **Algoritma Firefly**: Untuk mengoptimalkan proses clustering
    - **K-Means**: Untuk analisis clustering

    Aplikasi ini mendemonstrasikan penggunaan **K-Means clustering** dan **Algoritma Firefly + PCA** untuk mengoptimalkan proses clustering.

    """)


def show_preview_of_raw_data(df):
    st.title("Preview of Raw Data")

    st.subheader("First 5 Rows")
    st.write(df.head())

    st.subheader("Basic Statistics")
    st.write(df.describe())

    st.subheader("Data Types and Null Values")
    buffer = io.StringIO()
    df.info(buf=buffer)
    s = buffer.getvalue()
    st.text(s)

    st.subheader("Select Columns to Display")
    columns = st.multiselect(
        "Select Columns", df.columns.tolist(), default=df.columns.tolist()
    )
    st.write(df[columns])

    st.subheader("Filter Data")
    numeric_columns = df.select_dtypes(include=["float64", "int64"]).columns
    filter_column = st.selectbox("Select a Numeric Column to Filter", numeric_columns)
    min_value, max_value = st.slider(
        f"Select Range for {filter_column}",
        float(df[filter_column].min()),
        float(df[filter_column].max()),
        (float(df[filter_column].min()), float(df[filter_column].max())),
    )
    filtered_df = df[
        (df[filter_column] >= min_value) & (df[filter_column] <= max_value)
    ]
    st.write(filtered_df)


def show_firefly_and_pca():
    st.title("Apa itu Firefly dan PCA?")

    st.subheader("Algoritma Firefly")
    st.markdown("""
    **Algoritma Firefly** adalah algoritma optimisasi metaheuristik yang terinspirasi dari kunang-kunang dan dikembangkan oleh Xin-She Yang pada tahun 2008.
    Algoritma ini didasarkan pada pola berkedip dan perilaku kunang-kunang. Algoritma ini mengoptimalkan masalah dengan memindahkan "kunang-kunang"
    menuju individu yang lebih terang dan lebih menarik, yang dalam hal ini mewakili solusi yang lebih baik.

    **Fitur Utama:**
    - Menggunakan intensitas cahaya sebagai ukuran daya tarik.
    - Mensimulasikan perilaku kunang-kunang untuk menemukan solusi optimal.
    - Efektif untuk memecahkan masalah optimisasi yang kompleks seperti clustering.
    """)

    st.subheader("Principal Component Analysis (PCA)")
    st.markdown("""
    **Principal Component Analysis (PCA)** adalah teknik yang digunakan untuk pengurangan dimensi sambil mempertahankan 
    sebanyak mungkin variabilitas dalam dataset. PCA mengubah variabel asli menjadi kumpulan variabel baru
    yang disebut komponen utama, variable ini diurutkan berdasarkan jumlah varians yang mereka tangkap dari data asli.

    **Fitur Utama:**
    - Mengurangi dimensi dataset yang besar.
    - Membantu dalam memvisualisasikan struktur data berdimensi tinggi.
    - Meningkatkan efisiensi algoritma dengan mengurangi jumlah fitur input.

    PCA biasanya digunakan sebelum dilakukan clustering atau klasifikasi untuk meningkatkan perfoma dan mengurangi waktu komputasi.
    """)

    st.markdown("""
    **Sumber Tambahan:**
    - [Understanding Firefly Algorithm](https://www.baeldung.com/cs/firefly-algorithm)
    - [Explanation of Principal Components Analysis (PCA)](https://www.geeksforgeeks.org/principal-component-analysis-pca)
    """)


def show_eda(df):
    st.title("Exploratory Data Analysis (EDA)")

    st.subheader("Data Overview")
    st.write(df.head())

    st.subheader("Descriptive Statistics")
    st.write(df.describe())

    st.subheader("Sales Over Time")
    df["tanggal"] = pd.to_datetime(df["tanggal"], format="%d-%m-%Y")
    daily_sales = df.groupby("tanggal")["total_harga"].sum().reset_index()

    plt.figure(figsize=(10, 5))
    plt.plot(daily_sales["tanggal"], daily_sales["total_harga"])
    plt.title("Daily Sales Over Time")
    plt.xlabel("Date")
    plt.ylabel("Total Sales")
    st.pyplot(plt)

    st.subheader("Sales Distribution by Category")
    category_sales = df.groupby("kategori")["total_harga"].sum().reset_index()
    plt.figure(figsize=(10, 5))
    sns.barplot(x="kategori", y="total_harga", data=category_sales)
    plt.title("Total Sales by Category")
    plt.xlabel("Category")
    plt.ylabel("Total Sales")
    st.pyplot(plt)

    st.subheader("Most Popular Menu Items")
    popular_items = df["menu"].value_counts().reset_index()
    popular_items.columns = ["Menu Item", "Count"]
    st.write(popular_items.head(10))

    st.subheader("Order Type Analysis")
    order_type_counts = df["order"].value_counts().reset_index()
    order_type_counts.columns = ["Order Type", "Count"]

    plt.figure(figsize=(10, 5))
    sns.barplot(x="Order Type", y="Count", data=order_type_counts)
    plt.title("Order Type Distribution")
    plt.xlabel("Order Type")
    plt.ylabel("Count")
    st.pyplot(plt)

    st.subheader("Correlation Analysis")
    correlation_matrix = df[["jml_pembelian", "harga", "total_harga"]].corr()

    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", linewidths=0.5)
    plt.title("Correlation Heatmap")
    st.pyplot(plt)


df = load_data()

page = st.sidebar.radio(
    "Select a page:",
    ["Introduction", "Preview of Raw Data", "What is Firefly and PCA?", "EDA"],
)

if page == "Introduction":
    show_introduction()
elif page == "Preview of Raw Data":
    show_preview_of_raw_data(df)
elif page == "What is Firefly and PCA?":
    show_firefly_and_pca()
elif page == "EDA":
    show_eda(df)
