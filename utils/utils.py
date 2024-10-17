import pandas as pd
import streamlit as st

@st.cache_data
def load_data():
    # Load data from an Excel file and cache it.
    return pd.read_excel("data/data_penjualan_harian.xlsx")

@st.cache_data
def pre_process(df):
    # Convert 'tanggal' to datetime
    df["tanggal"] = pd.to_datetime(
        df["tanggal"], format="%d-%m-%Y"
    )  # Adjust the format if needed

    # Convert 'waktu_pembelian' to time
    df["waktu_pembelian"] = pd.to_datetime(df["waktu_pembelian"], format="%H:%M")

    #  # Extract hour and minute
    df["hour"] = df["waktu_pembelian"].dt.hour
    df["minute"] = df["waktu_pembelian"].dt.minute
    # Extract hour and minute
    df.insert(2, 'month', df['tanggal'].dt.month)
    df.insert(3, 'day', df['tanggal'].dt.day)

    # Drop 'tanggal' and 'waktu_pembelian' columns
    df.drop(columns=["waktu_pembelian","tanggal", ], inplace=True)

    result = df.groupby(['month', 'day', 'hour', 'minute', 'order']).agg(
        jml_pembelian=('jml_pembelian', 'sum'),
        total_harga=('total_harga', 'sum'),
        kategori_Coffee=('total_harga', lambda x: x[df.loc[x.index, 'kategori'] == 'Coffee'].sum()),
        kategori_Food=('total_harga', lambda x: x[df.loc[x.index, 'kategori'] == 'Food'].sum()),
        kategori_Non_Coffee=('total_harga', lambda x: x[df.loc[x.index, 'kategori'] == 'Non-Coffee'].sum()),
        jml_Coffee=('jml_pembelian', lambda x: x[df.loc[x.index, 'kategori'] == 'Coffee'].sum()),
        jml_Food=('jml_pembelian', lambda x: x[df.loc[x.index, 'kategori'] == 'Food'].sum()),
        jml_Non_Coffee=('jml_pembelian', lambda x: x[df.loc[x.index, 'kategori'] == 'Non-Coffee'].sum())
    ).reset_index()

    # Fix column names if needed
    result.columns = ['month', 'day', 'hour', 'minute', 'order', 'jml_pembelian', 'total_harga', 
                    'kategori_Coffee', 'kategori_Food', 'kategori_Non_Coffee', 
                    'jml_Coffee', 'jml_Food', 'jml_Non_Coffee']
    
    df = result.copy()
    
    df = df.drop(['month', 'day', 'order'], axis=1)
    # Convert hours and minutes to radians
    # df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    # df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    # df['minute_sin'] = np.sin(2 * np.pi * df['minute'] / 60)
    # df['minute_cos'] = np.cos(2 * np.pi * df['minute'] / 60)

    # Drop the original hour and minute columns
    df = df.drop(['hour', 'minute'], axis=1)
    
    # Calculate average price per item
    df['average_price'] = df['total_harga'] / df['jml_pembelian']
    
    # Calculate category spending ratios
    df['ratio_Coffee'] = df['kategori_Coffee'] / df['total_harga']
    df['ratio_Food'] = df['kategori_Food'] / df['total_harga']
    df['ratio_Non_Coffee'] = df['kategori_Non_Coffee'] / df['total_harga']
    df = df.drop(columns=[ 'jml_Coffee', 'jml_Food', 'jml_Non_Coffee'])
    
    return df