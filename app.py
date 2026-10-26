import os
import re
import platform
from collections import Counter
import streamlit as st
from PIL import Image
from PyPDF2 import PdfReader

# Imports actualizados para evitar errores de deprecación
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 1. Configuración de la página y diseño
st.set_page_config(
    page_title="RAG PDF Analyzer | Miguel Rueda",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos visuales personalizados (CSS)
st.markdown("""
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #2C3E50;
        margin-bottom: 0px;
    }
    .sub-title {
        font-size: 1.1rem;
        color: #7F8C8D;
        margin-bottom: 20px;
    }
    .stMetric {
        background-color: #F8F9F9;
        padding: 10px;
        border-radius: 8px;
        border: 1px solid #E5E7E9;
    }
    </style>
""", unsafe_allow_html=unsafe_allow_ok=True)

# 2. Encabezado y Marca Personal
st.markdown('<p class="main-title">Análisis Inteligente de PDFs (RAG) 💬</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Herramienta de procesamiento de documentos y consulta interactiva con IA</p>', unsafe_allow_html=True)

# Cargar imagen de branding o banner si existe
try:
    image = Image.open('Chat_pdf.png')
    st.image(image, width=320)
except Exception:
    pass

# 3. Barra Lateral (Sidebar)
with st.sidebar:
    st.header("⚙️ Configuración")
    
    ke = st.text_input('Clave API de Google Gemini', type="password", help="Obtén tu clave gratuita en Google AI Studio")
    if ke:
        os.environ['GOOGLE_API_KEY'] = ke
    else:
        st.warning("🔑 Ingresa tu clave para habilitar el motor RAG.")

    st.divider()
    
    st.subheader("👨‍💻 Desarrollado por")
    st.write("**Miguel Rueda**")
    st.caption("Pregrado en Diseño Interactivo - EAFIT")
    st.caption(f"Python Environment: v{platform.python_version()}")

# 4. Carga de Archivos
st.subheader("1. Carga de Documento")
pdf = st.file_uploader("Selecciona un archivo PDF para analizar", type="pdf")

if pdf is not None:
    if not ke:
        st.info("👈 Ingresa tu API Key en la barra lateral para iniciar el procesamiento del PDF.")
    else:
        try:
            # Extracción del texto
            pdf_reader = PdfReader(pdf)
            text = ""
            for page in pdf_reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"

            # Métricas rápidas del documento
            st.success("¡PDF cargado e indexado con éxito!")
            col1, col2, col3 = st.columns(3)
            col1.metric("Páginas Totales", len(pdf_reader.pages))
            col2.metric("Caracteres Extraídos", f"{len(text):,}")
            col3.metric("Palabras Aprox.", f"{len(text.split()):,}")

            st.divider()

            # APORTE ORIGINAL: Análisis de Frecuencia de Palabras Clave
            st.subheader("2. Palabras Clave y Temáticas Relevantes")
            words = re.findall(r'\b[a-zA-ZáéíóúÁÉÍÓÚñÑ]{5,}\b', text.lower())
            stopwords = {"sobre", "entre", "hasta", "desde", "donde", "cuando", "porque", "esta", "este", "estos", "mismo", "tiene"}
            filtered_words = [w for w in words if w not in stopwords]
            common_words = Counter(filtered_words).most_common(5)

            k_cols = st.columns(len(common_words))
            for idx, (word, count) in enumerate(common_words):
                k_cols[idx].metric(label=word.capitalize(), value=f"{count} menciones")

            st.divider()

            # Preparación del RAG (Chunks + Embeddings)
            text_splitter = CharacterTextSplitter(
                separator="\n",
                chunk_size=700,
                chunk_overlap=100,
                length_function=len
            )
            chunks = text_splitter.split_text(text)

            with st.spinner("Indexando vectores con Gemini Embeddings..."):
                embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
                knowledge_base = FAISS.from_texts(chunks, embeddings)
                llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.2)

            # 5. Interfaz de Consultas
            st.subheader("3. Consulta Interactiva sobre el PDF")
            user_question = st.text_input("Haz una pregunta específica sobre el contenido del archivo:")

            if user_question:
                with st.spinner("Analizando respuesta en la base de conocimiento..."):
                    docs = knowledge_base.similarity_search(user_question, k=3)
                    
                    # Construcción de la cadena moderna LCEL
                    context_text = "\n\n".join([doc.page_content for doc in docs])
                    prompt = ChatPromptTemplate.from_template(
                        "Responde a la siguiente pregunta basándote únicamente en el contexto provisto.\n"
                        "Si la respuesta no se encuentra en el texto, indícalo educadamente.\n\n"
                        "Contexto:\n{context}\n\n"
                        "Pregunta: {question}"
                    )
                    chain = prompt | llm | StrOutputParser()
                    response = chain.invoke({"context": context_text, "question": user_question})

                    # Mostrar Respuesta
                    st.markdown("#### 🤖 Respuesta del Asistente:")
                    st.info(response)

                    # Mostrar Contexto Utilizado
                    with st.expander("🔍 Ver fragmentos de texto recuperados (RAG Context)"):
                        for i, doc in enumerate(docs):
                            st.caption(f"**Fragmento {i+1}:**")
                            st.text(doc.page_content
