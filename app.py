import os
import re
import platform
from collections import Counter
import streamlit as st
from PIL import Image
from PyPDF2 import PdfReader

# Componentes de LangChain v0.3+
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 1. Configuración de página
st.set_page_config(
    page_title="RAG PDF Studio | Miguel Rueda",
    page_icon="⚡",
    layout="wide"
)

# 2. CSS Seguro (sin romper clases internas de Streamlit)
st.markdown("""
    <style>
    .header-container {
        background: linear-gradient(135deg, #0A2540 0%, #1A365D 100%);
        padding: 2rem;
        border-radius: 12px;
        border-left: 8px solid #F6AD55;
        color: white;
        margin-bottom: 2rem;
    }
    .header-title {
        font-size: 2.2rem;
        font-weight: 800;
        margin: 0;
        color: #FFFFFF;
    }
    .yellow-txt {
        color: #ECC94B;
    }
    .header-sub {
        color: #90CDF4;
        font-size: 1rem;
        margin-top: 5px;
    }
    div[data-testid="stMetric"] {
        background-color: #EBF8FF !important;
        border: 2px solid #BEE3F8 !important;
        border-radius: 10px !important;
        padding: 12px !important;
    }
    </style>
""", unsafe_allow_html=True)

# 3. Encabezado principal
st.markdown("""
    <div class="header-container">
        <h1 class="header-title">⚡ RAG PDF <span class="yellow-txt">Studio</span></h1>
        <p class="header-sub">Análisis inteligente de documentos con IA y búsqueda vectorial</p>
    </div>
""", unsafe_allow_html=True)

# Cargar imagen opcional
try:
    image = Image.open('Chat_pdf.png')
    st.image(image, width=250)
except Exception:
    pass

# 4. Barra Lateral
with st.sidebar:
    st.header("⚙️ Configuración")
    
    ke = st.text_input('API Key de Google Gemini', type="password", help="Genera tu llave gratuita en Google AI Studio")
    if ke:
        os.environ['GOOGLE_API_KEY'] = ke
        st.success("🔑 Key conectada")
    else:
        st.warning("⚠️ Ingresa tu API Key para activar el bot.")

    st.divider()
    st.write("**Desarrollado por:**")
    st.write("Miguel Rueda")
    st.caption("Diseño Interactivo • EAFIT")
    st.caption(f"Python v{platform.python_version()}")

# 5. Carga y Procesamiento de PDF
st.subheader("📁 1. Cargar Documento PDF")
pdf = st.file_uploader("Sube tu archivo PDF aquí", type="pdf")

if pdf is not None:
    if not ke:
        st.warning("👈 Por favor, ingresa tu API Key en la barra lateral para continuar.")
    else:
        try:
            pdf_reader = PdfReader(pdf)
            text = ""
            for page in pdf_reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"

            st.divider()
            st.subheader("📊 Métricas del Documento")
            
            c1, c2, c3 = st.columns(3)
            c1.metric("Páginas Totales", len(pdf_reader.pages))
            c2.metric("Caracteres Extraídos", f"{len(text):,}")
            c3.metric("Palabras Procesadas", f"{len(text.split()):,}")

            # Análisis de Palabras Clave
            st.divider()
            st.subheader("🔑 Palabras Más Frecuentes")
            
            words = re.findall(r'\b[a-zA-ZáéíóúÁÉÍÓÚñÑ]{5,}\b', text.lower())
            stopwords = {"sobre
