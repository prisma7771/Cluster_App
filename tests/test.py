import joblib
import numpy as np
import pytest
import pandas as pd
import io
import sys
sys.path.insert(1,"E:\\Programming\\Python\\JupyterNotebook\\Clustering\\Tugas_Akhir\\App")

from utils.utils import pre_process

kmeans_model = joblib.load("../data/kmeans_model_pca_firefly.joblib")

# Dummy CSV file upload
def get_valid_csv_file():
    # Simulating a CSV with correct format and sample data

    data = (
        "tanggal,waktu_pembelian,jml_pembelian,harga,kategori,menu,total_harga,order\n"
        "01-04-2024,12:30,2,10000,Coffee,Latte,20000,12345\n"
        "02-04-2024,13:00,1,15000,Food,Strawberry Waffle,15000,12346\n"
    )

    return io.StringIO(data)  # Returns file-like object



@pytest.fixture
def valid_csv():
    return get_valid_csv_file()

# Test case for valid file upload
def test_file_upload_valid(valid_csv):
    # Simulating file upload in Streamlit
    uploaded_file = valid_csv
    data_uploaded = pd.read_csv(uploaded_file)

    # Check if columns match
    expected_columns = [
        "tanggal",
        "waktu_pembelian",
        "jml_pembelian",
        "harga",
        "kategori",
        "menu",
        "total_harga",
        "order",
    ]
    assert list(data_uploaded.columns.tolist()) == expected_columns, "Columns mismatch"

    # Test the pre-processing function
    processed_data = pre_process(data_uploaded)
    assert not processed_data.empty, "Processed data is empty"


# Test case for invalid file format
@pytest.mark.xfail
def test_file_upload_invalid_format():
    # Simulating invalid CSV (wrong format, missing columns)
    data = """date,amount,category,menu,total_price
    2024-12-01,10000,Coffee,Latte,20000"""
    invalid_file = io.StringIO(data)
    with pytest.raises(
        ValueError, match="Invalid COlumn"
    ):  # Assuming pre_process throws a ValueError for format issues
        data_uploaded = pd.read_csv(invalid_file)
        processed_data = pre_process(data_uploaded)
        assert processed_data.empty, "Invalid data should raise an error"


# # Test the predictions after file upload
def test_predict_clusters_after_upload(valid_csv):
    uploaded_file = valid_csv
    data_uploaded = pd.read_csv(uploaded_file)

    processed_data = pre_process(data_uploaded)
    predictions = kmeans_model.predict(
        processed_data
    )  # Assuming model is loaded properly
    print(type(predictions))
    # Check if predictions were made and data is labeled correctly
    assert len(predictions) == len(data_uploaded), "Prediction count mismatch"
    assert np.issubdtype(predictions.dtype, np.integer), "Cluster is not integer"
