import os
import streamlit as st
from langchain_community.vectorstores import Weaviate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms import HuggingFaceHub
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
# Assuming this is where rag_chain is defined

# Environment variables
WEAVIATE_CLUSTER="https://g5ydc6potuimpatrebbokg.c0.europe-west3.gcp.weaviate.cloud"
WEAVIATE_API_KEY="***************************************"

# Initialize Weaviate client
import weaviate
client = weaviate.connect_to_wcs(
    cluster_url=WEAVIATE_CLUSTER,  # Replace with your WCD URL
    auth_credentials=weaviate.auth.AuthApiKey(
       WEAVIATE_API_KEY
    ),  # Replace with your WCD key
)


# Specify the path to the folder containing PDF files
pdf_folder = r"C:\Users\Ayush\Downloads\IndianCuisine"  # Update with your folder path

# List all PDF files in the folder
pdf_files = [os.path.join(pdf_folder, file) for file in os.listdir(pdf_folder) if file.endswith(".pdf")]

# List to store the loaded data
all_data = []

# Loop through each file and load the data
for pdf_file in pdf_files:
    loader = PyMuPDFLoader(pdf_file)
    data = loader.load()
    all_data.extend(data)

# Split documents into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
docs = text_splitter.split_documents(all_data)

# Create vector database in Weaviate
embeddings_model_name = "sentence-transformers/all-MiniLM-L6-v2"
embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)
vector_db = Weaviate.from_documents(docs, embeddings, client=client, by_text=False)

# Define Chat Prompt Template
template = """
You are a question-answering agent. You should answer questions based on the context provided and
keep the answer precise and concise. You may add some of your own knowledge to make the answer more comprehensive.
Give an answer in no more than 10 sentences.
Question: {Query}
Context: {contexts}
Answer:
"""
prompt = ChatPromptTemplate.from_template(template)

# Define Hugging Face Model
model = HuggingFaceHub(
    huggingfacehub_api_token="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",
    repo_id="mistralai/Mistral-7B-Instruct-v0.3",
    model_kwargs={"temperature": 0.5, "max_length": 500}
)

# Define Output Parser
output_parser = StrOutputParser()

# Define Runnable Passthrough for Query
retriever = vector_db.as_retriever()
rag_chain = (
    {"contexts": retriever, "Query": RunnablePassthrough()}
    | prompt
    | model
    | output_parser
)

# Set up the Streamlit page configuration and styling
st.set_page_config(page_title="FlavorFolklore", page_icon="🍛")

# Custom CSS for centering text and adding styles
st.markdown("""
    <style>
    .centered-title {
        text-align: center;
        font-size: 48px;
        color: #8B0000;
        font-family: 'Trebuchet MS', sans-serif;
    }
    .centered-tagline {
        text-align: center;
        font-size: 24px;
        color: #FF6347;
        font-family: 'Trebuchet MS', sans-serif;
        margin-top: -20px;
    }
    .input-box {
        margin-top: 50px;
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)

# Centered title and tagline
st.markdown('<h1 class="centered-title">🍛 FlavorFolklore 🍛</h1>', unsafe_allow_html=True)
st.markdown('<h2 class="centered-tagline">Tales from the Indian Kitchen</h2>', unsafe_allow_html=True)

# Main content
st.markdown("""
Welcome to FlavorFolklore!
Get the perfect Indian recipes tailored to your needs.
Simply ask a question, and we'll provide a precise and concise answer, drawing from a rich context of Indian culinary traditions.
""", unsafe_allow_html=True)

# Input box and button
query = st.text_input("Ask about your favorite Indian dish, spice, or recipe:", placeholder="e.g., What are the ingredients for biryani?", key="input_box")
if st.button("Discover Recipe 🍲"):
    if query:
        result = rag_chain({"Query": query})
        st.write("### Unlock the Magic of Your Dish!")
        st.write(result)
    else:
        st.write("Please enter a question related to FlavorFolklore 🌶️.")
