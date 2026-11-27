"""PropPilot Streamlit multipage application entrypoint."""

from __future__ import annotations

import streamlit as st

pages = [
    st.Page("app_pages/home.py", title="Home", default=True),
    st.Page("pages/1_Simulate.py", title="Simulate"),
    st.Page("pages/2_Optimise.py", title="Optimise"),
    st.Page("pages/3_Funded_Projection.py", title="Funded Projection"),
    st.Page("pages/4_Rules_Lab.py", title="Rules Lab"),
    st.Page("pages/5_AI_Copilot.py", title="AI Copilot"),
    st.Page("pages/6_About_Methodology.py", title="About & Methodology"),
]

pg = st.navigation(pages)
pg.run()
