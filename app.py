import streamlit as st
st.set_page_config(page_title="Peramalan")

peramalanPage = st.Page("page/peramalan.py", title="Peramalan", icon=":material/analytics:")
hasilPeramalanPage = st.Page("page/hasilPeramalan.py", title="Hasil Peramalan", icon=":material/dashboard:")


pg = st.navigation([peramalanPage, hasilPeramalanPage])
pg.run()
