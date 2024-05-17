import os
import streamlit as st
from datetime import datetime, timezone
from google.cloud import aiplatform, bigquery
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
from typing import List
from langchain.text_splitter import CharacterTextSplitter
from langchain_google_vertexai import VertexAI
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Chroma

# Setting environment variables for Google Cloud Platform
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/Users/abhis_m3/Desktop/LLM Cyber/service_acc.json"
project_id = "d5assistant"
location = "us-central1"

# Initialize Google Vertex AI and BigQuery Client
aiplatform.init(project=project_id, location=location)
bq_client = bigquery.Client()

# Define schema and create dataset/table if not exists
def create_bigquery_dataset_table(dataset_id, table_id, schema):
    dataset_ref = bq_client.dataset(dataset_id)
    try:
        bq_client.get_dataset(dataset_ref)
    except Exception:
        dataset = bigquery.Dataset(dataset_ref)
        dataset.location = "US"
        bq_client.create_dataset(dataset, exists_ok=True)

    table_ref = dataset_ref.table(table_id)
    try:
        bq_client.get_table(table_ref)
    except Exception:
        table = bigquery.Table(table_ref, schema=schema)
        bq_client.create_table(table, exists_ok=True)

schema = [
    bigquery.SchemaField("question", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("answer", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("timestamp", "TIMESTAMP", mode="REQUIRED"),
]

# Create dataset and table
dataset_id = "questions_answers"
table_id = "conversations"
create_bigquery_dataset_table(dataset_id, table_id, schema)

# Sentence Transformers for Embeddings
class TransformerEmbeddings:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(texts, convert_to_tensor=False).tolist()

    def embed_query(self, text: str) -> List[float]:
        return self.model.encode(text, convert_to_tensor=False).tolist()

# Streamlit UI setup
st.set_page_config(page_title="Mr.DOC", page_icon="📄")
st.header("Ask Your PDF 📄")

if "history" not in st.session_state:
    st.session_state.history = []

# Display conversation history as interactive chat
for exchange in reversed(st.session_state.history):
    with st.chat_message("user"):
        st.write(f"Q: {exchange['question']}")
    with st.chat_message("assistant"):
        st.write(f"A: {exchange['answer']}")

# PDF Upload
pdf = st.file_uploader("Upload your PDF", type="pdf")
if pdf:
    pdf_reader = PdfReader(pdf)
    text = "".join([page.extract_text() for page in pdf_reader.pages if page.extract_text()])
    if text:
        text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_text(text)

        embeddings = TransformerEmbeddings()
        chroma_index = Chroma.from_texts(chunks, embeddings)
        retriever = chroma_index.as_retriever()

        # Setup the model with maximum parameters for Gemini Pro
        llm = VertexAI(
            model_name="gemini-pro",
            temperature=0.7,  # Adjust as necessary
            max_output_tokens=2048,  # Maximum token size for Gemini Pro
            top_p=1.0
        )
        qa_chain = RetrievalQA.from_chain_type(llm, chain_type="stuff", retriever=retriever)

        if prompt := st.chat_input("Ask a question about your PDF"):
            response = qa_chain.invoke({"query": prompt})  # Updated to use .invoke()
            response_str = response.get("result", "") if response else ""
            record = {
                "question": prompt,
                "answer": response_str,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            st.session_state.history.append(record)

            # Save to BigQuery
            table_ref = f"{project_id}.{dataset_id}.{table_id}"
            try:
                bq_client.insert_rows_json(table_ref, [record])
            except Exception as e:
                st.error(f"Failed to insert record into BigQuery: {e}")
    else:
        st.error("Failed to extract text from PDF.")
