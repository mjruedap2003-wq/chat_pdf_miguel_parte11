import os
import platform
import streamlit as st
from PIL import Image
from PyPDF2 import PdfReader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from collections import Counter
import re

# Configuración de la página
st.set_page_config(page_title="Analizador RAG de PDFs", page_icon="📄", layout="wide")

st.title('Generador RAG + Analizador de PDFs 📄🤖')
st.caption(f"Versión de Python: {platform.python_version()}")

# Cargar imagen si existe
try:
    image = Image.open('Chat_pdf.png')
    st.image(image, width=300)
except Exception:
    pass

# Barra lateral
with st.sidebar:
    st.header("🔑 Configuración")
    api_key = st.text_input('Ingresa tu API Key de Google Gemini', type="password")
    st.markdown("[Obtén tu API Key gratuita aquí](https://aistudio.google.com/)")
    
    st.divider()
    st.subheader("💡 Características Originales")
    st.write("• Resumen Ejecutivo Automático")
    st.write("• Extracción de Palabras Clave")
    st.write("• Búsqueda y RAG conversacional")

if api_key:
    os.environ['GOOGLE_API_KEY'] = api_key

pdf = st.file_uploader("Carga tu archivo PDF aquí", type="pdf")

if pdf is not None and api_key:
    try:
        # 1. Extracción de texto
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"

        st.info(f"📊 **Métricas del documento:** {len(pdf_reader.pages)} páginas | {len(text)} caracteres")

        # =========================================================
        # APORTE ORIGINAL 1: Análisis estadístico de Palabras Clave
        # =========================================================
        st.subheader("📌 Análisis de Palabras Clave del PDF")
        # Limpiar texto para obtener palabras más relevantes
        words = re.findall(r'\b[a-zA-ZáéíóúÁÉÍÓÚñÑ]{4,}\b', text.lower())
        stopwords = {"para", "como", "esta", "este", "entre", "sobre", "donde", "cuando", "desde", "hasta", "este", "estos", "estas"}
        filtered_words = [w for w in words if w not in stopwords]
        common_words = Counter(filtered_words).most_common(8)

        cols = st.columns(len(common_words))
        for idx, (word, count) in enumerate(common_words):
            cols[idx].metric(label=word.capitalize(), value=f"{count}x")

        # 2. División del texto en fragmentos (Chunks)
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=800,
            chunk_overlap=100,
            length_function=len
        )
        chunks = text_splitter.split_text(text)

        # 3. Embeddings e Índice vectorial con Gemini
        with st.spinner("Creando base de conocimiento vectorial..."):
            embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
            knowledge_base = FAISS.from_texts(chunks, embeddings)

        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)

        # =========================================================
        # APORTE ORIGINAL 2: Resumen Ejecutivo Automático
        # =========================================================
        with st.expander("✨ Generar Resumen Ejecutivo Automático del Documento", expanded=False):
            if st.button("Generar Resumen e Ideas Principales"):
                with st.spinner("Sintetizando el documento..."):
                    docs_summary = knowledge_base.similarity_search("resumen introduccion objetivos conclusiones", k=4)
                    chain_summary = load_qa_chain(llm, chain_type="stuff")
                    prompt_resumen = "Haz un resumen ejecutivo corto de este documento en 3 viñetas y extrae los 3 objetivos principales."
                    summary_response = chain_summary.run(input_documents=docs_summary, question=prompt_resumen)
                    st.markdown(summary_response)

        st.divider()

        # 4. Preguntas y Respuestas (RAG)
        st.subheader("💬 Haz una pregunta específica sobre el PDF")
        user_question = st.text_input("Escribe tu pregunta:", placeholder="Ej: ¿Cuáles son las conclusiones principales?")

        if user_question:
            with st.spinner("Buscando en el documento..."):
                docs = knowledge_base.similarity_search(user_question, k=3)
                chain = load_qa_chain(llm, chain_type="stuff")
                response = chain.run(input_documents=docs, question=user_question)

                st.markdown("### 🤖 Respuesta del Asistente:")
                st.write(response)

                # Mostrar fuentes/contextos recuperados
                with st.expander("Ver fragmentos del PDF utilizados para responder"):
                    for i, doc in enumerate(docs):
                        st.write(f"**Fragmento {i+1}:**")
                        st.caption(doc.page_content)

    except Exception as e:
        st.error(f"Error al procesar el PDF: {str(e)}")

elif pdf is not None and not api_key:
    st.warning("⚠️ Ingresa tu API Key de Google Gemini en la barra lateral para procesar el archivo.")
else:
    st.info("👈 Ingresa tu API Key y carga un archivo PDF para comenzar.")
