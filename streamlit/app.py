from pathlib import Path

import streamlit as st

# create the title
file = Path(__file__)
rootdir = file.parents[1]
long_description = (rootdir / "README.md").read_text()
st.markdown(body=long_description)
