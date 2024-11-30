import streamlit as st
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer, util
import torch
import re

# Função para carregar o modelo de embeddings
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_embedding_model()

# Função para extrair texto de um arquivo PDF
def extract_text_from_pdf(pdf_file):
    reader = PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

# Função para preprocessar e dividir o texto em sentenças
def preprocess_text(text):
    # Remove quebras de linha e espaços extras
    text = text.replace('\n', ' ').strip()
    # Divide o texto em sentenças usando expressões regulares
    sentences = re.split(r'(?<=[.!?])\s+', text)
    # Filtra sentenças muito curtas
    sentences = [sentence for sentence in sentences if len(sentence.split()) > 5]
    return sentences

# Função para gerar embeddings para uma lista de sentenças
def generate_embeddings(sentences):
    embeddings = model.encode(sentences, convert_to_tensor=True)
    return embeddings

# Função para encontrar as sentenças mais relevantes para uma consulta
def find_relevant_sentences(query, sentences, embeddings, top_k=5):
    query_embedding = model.encode(query, convert_to_tensor=True)
    # Calcula a similaridade coseno entre a consulta e os embeddings das sentenças
    cosine_scores = util.cos_sim(query_embedding, embeddings)[0]
    # Recupera as top_k sentenças mais similares
    top_results = torch.topk(cosine_scores, k=top_k)
    relevant_sentences = []
    for score, idx in zip(top_results.values, top_results.indices):
        relevant_sentences.append((sentences[idx], score.item()))
    return relevant_sentences

# Interface da aplicação Streamlit
def main():
    st.title("Assistente Conversacional Baseado em LLM")

    # Upload do arquivo PDF
    uploaded_file = st.file_uploader("Escolha um arquivo PDF", type=["pdf"])
    
    # Variáveis de estado para armazenar sentenças e embeddings
    if 'sentences' not in st.session_state:
        st.session_state.sentences = []
    if 'embeddings' not in st.session_state:
        st.session_state.embeddings = None

    if uploaded_file:
        with st.spinner("Processando o documento..."):
            # Extrai o texto do PDF
            text = extract_text_from_pdf(uploaded_file)
            # Preprocessa o texto e divide em sentenças
            sentences = preprocess_text(text)
            # Gera embeddings para as sentenças
            embeddings = generate_embeddings(sentences)
            # Armazena nas variáveis de estado
            st.session_state.sentences = sentences
            st.session_state.embeddings = embeddings

    # Campo de pergunta
    if st.session_state.embeddings is not None:
        query = st.text_input("Digite sua pergunta:")
        if query:
            with st.spinner("Processando sua pergunta..."):
                relevant_sentences = find_relevant_sentences(query, st.session_state.sentences, st.session_state.embeddings)
            st.write("**Respostas mais relevantes:**")
            for sentence, score in relevant_sentences:
                st.markdown(f"- **Relevância:** {score:.4f}")
                st.write(sentence)
    else:
        st.info("Por favor, faça o upload de um documento PDF para começar.")

if __name__ == "__main__":
    main()
