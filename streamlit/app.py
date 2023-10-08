from pathlib import Path

import streamlit as st

# create the title
file = Path(__file__)
rootdir = file.parents[1]
title = " ".join(rootdir.name.split("-")).title()
assert title.endswith("Lab"), title
# create the UI components
st.title(title)
long_description = (rootdir / "README.md").read_text()
st.markdown(body=long_description)
