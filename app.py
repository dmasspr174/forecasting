import streamlit as st
st.set_page_config(page_title="Peramalan")

peramalannPage = st.Page("page/peramalann.py", title="Peramalan", icon=":material/analytics:")
hasilPeramalanPage = st.Page("page/hasilPeramalan.py", title="Hasil Peramalan", icon=":material/dashboard:")


pg = st.navigation([peramalannPage, hasilPeramalanPage])
pg.run()
