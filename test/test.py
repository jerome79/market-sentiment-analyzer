import os
import sys
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

# -------- repo-root import fix --------
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# optional plotting helpers (fallbacks defined below if not present)


# -------------------- App config --------------------
load_dotenv()
st.set_page_config(page_title="Market Sentiment Analyzer", layout="wide")
st.set_option("client.showErrorDetails", True)


# -------------------- UI --------------------

# -------------------- Sidebar debug --------------------
with st.sidebar:
    st.header("🔧 Debug")
    st.caption("Use this to verify inputs and code path.")
    st.write("CWD:", os.getcwd())
    st.write("NEWS_CSV_DIR:", os.getenv("NEWS_CSV_DIR", "(unset)"))
    st.write("SECTOR_MAP_CSV:", os.getenv("SECTOR_MAP_CSV", "(unset)"))
    st.write("SENTIMENT_MODEL:", os.getenv("SENTIMENT_MODEL", "(unset)"))
    st.write("Python:", sys.version.split()[0])

st.title("📈 Market Sentiment Analyzer")
