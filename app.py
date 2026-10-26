import os
import re
import platform
from collections import Counter
import streamlit as st
from PIL import Image
from PyPDF2 import PdfReader

# Componentes actualizados de LangChain (Sintaxis LCEL v0.3+)
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 1. Configuración de la página
st.set_page_config(
    page_title="RAG PDF Studio | Miguel Rueda",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. CSS Personalizado: Estilo Azul & Amarillo
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    .header-box {
        background: linear-gradient(135deg, #0A2540 0%, #1A365D 100%);
        padding: 2.2rem 2rem;
        border-radius: 16px;
        border-left: 8px solid #F6AD55;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 10px 25px -5px rgba(10, 37, 64, 0.25);
    }
    
    .header-title {
        font-size: 2.3rem;
        font-weight: 800;
        margin: 0;
        color: #FFFFFF;
        letter-spacing: -0.5px;
    }
    
    .header-subtitle {
        font-size: 1.05rem;
        color: #90CDF4;
        margin-top: 8px;
        margin-bottom: 0;
    }

    .yellow-accent {
        color: #ECC94B;
        font-weight: 700;
    }

    div[data-testid="stMetric"] {
        background: #EBF8FF;
        border: 2px solid #BEE3F8;
        border-radius: 12px;
        padding: 15px;
    }

    div[data-testid="stMetricLabel"] {
        color: #2B6CB0 !important;
        font-weight: 600;
    }

    div[data-testid="stMetricValue"] {
        color: #0A2540 !important;
        font-weight: 800;
    }

    .custom-info-box {
        background-color: #FEFCBF;
        border-left: 5px solid #D69E2E;
        color: #744210;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        font-weight: 600;
    }

    .stButton>button {
        background: linear-gradient(90deg, #3182CE 0%, #2B6CB0 100%);
        color: white;
        font-weight: 700;
        border: none;
        border-radius: 10px;
        padding: 0.6rem 1.5rem;
    }

    .stButton>button:hover {
        background: linear-gradient(90deg, #D69E2E 0%, #B7791F 100%);
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

#
