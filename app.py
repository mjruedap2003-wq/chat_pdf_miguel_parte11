import os
import re
import platform
from collections import Counter
import streamlit as st
from PIL import Image
from PyPDF2 import PdfReader

# Imports de LangChain LCEL
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

# 2. CSS Personalizado: Tema Azul + Acentos Amarillos & Estilo Dinámico
st.markdown("""
    <style>
    /* Fondo general e importación de fuente */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    /* Ocultar barra superior por defecto */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Contenedor del Título Principal */
    .header-box {
        background: linear-gradient(135deg, #0A2540 0%, #1A365D 100%);
        padding: 2.5rem 2rem;
        border-radius: 16px;
        border-left: 8px solid #F6AD55; /* Amarillo cálido */
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 10px 25px -5px rgba(10, 37, 64, 0.25);
    }
    
    .header-title {
        font-size: 2.4rem;
        font-weight: 800;
        margin: 0;
        color: #FFFFFF;
        letter-spacing: -0.5px;
    }
    
    .header-subtitle {
        font-size: 1.1rem;
        color: #90CDF4; /* Azul claro */
        margin-top: 8px;
        margin-bottom: 0;
    }

    /* Destacados en amarillo */
    .yellow-accent {
        color: #ECC94B;
        font-weight: 7
