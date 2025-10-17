# Import necessary libraries
import fitz  # PyMuPDF
from transformers import pipeline
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import streamlit as st


# Function to extract text from a PDF using PyMuPDF
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text("text")  # Extract plain text from each page
    return text

# Initialize the Granite model for text generation
pipe = pipeline("text-generation", model="ibm-granite/granite-3.2-2b-instruct")

# Function to create embeddings for PDF text chunks using SentenceTransformer
def create_embeddings(pdf_text):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    return model.encode(pdf_text, convert_to_tensor=True)

# Function to create a FAISS index for the embeddings
def create_faiss_index(embeddings):
    faiss_index = faiss.IndexFlatL2(embeddings.shape[1])
    faiss_index.add(np.array(embeddings))
    return faiss_index

# Function to get the most relevant chunk for a query using FAISS
def get_relevant_chunk(query, pdf_text, faiss_index):
    query_embedding = model.encode([query])
    _, indices = faiss_index.search(query_embedding, k=1)
    return pdf_text[indices[0][0]]  # Return the most relevant chunk of text

# Function to generate an answer using the Granite model
def generate_answer(query, relevant_chunk):
    response = pipe([{"role": "user", "content": query + " " + relevant_chunk}])
    return response[0]['generated_text']

# Streamlit interface setup
def run_streamlit_app():
    st.title("StudyMate: AI-Powered Q&A System")

    # File Upload
    uploaded_file = st.file_uploader("Upload your PDF", type="pdf")
    if uploaded_file is not None:
        # Extract text from the uploaded PDF
        text = extract_text_from_pdf(uploaded_file)
        pdf_text = text.split('\n')  # Split text into chunks (could be pages, paragraphs, etc.)

        # Create embeddings and FAISS index
        embeddings = create_embeddings(pdf_text)
        faiss_index = create_faiss_index(embeddings)

        st.text_area("Extracted Text", text[:2000])  # Show first 2000 characters of extracted text

    # User Query Input
    question = st.text_input("Ask a question")
    if question:
        # Find the most relevant chunk
        relevant_chunk = get_relevant_chunk(question, pdf_text, faiss_index)

        # Generate the answer using the Granite model
        answer = generate_answer(question, relevant_chunk)
        st.write(answer)

# Run the Streamlit app
run_streamlit_app()