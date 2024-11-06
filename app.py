import streamlit as st


pages = [
    st.Page("routes/home.py", title="HomePage"),
    st.Page("routes/pre_processing.py", title="Pre-Processing Data"),
    st.Page("routes/comparison.py", title="Comparison"),
    st.Page("routes/cluster.py", title="Cluster Yourself!"),
]


pg = st.navigation(pages)
st.set_page_config(page_title="Main Menu")
pg.run()
