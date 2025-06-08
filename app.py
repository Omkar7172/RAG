import streamlit as st
import PyPDF2

st.header('A RAG App')
uploaded_file = r"C:\Users\Dell\Downloads\Omkar_Hase_Resume (3).pdf"

text = ""
pdf_reader = PyPDF2.PdfReader(uploaded_file)
for page in pdf_reader.pages:
    text +=page.extract_text() + "\n"


groqapi = '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'

from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)

text_chunks = splitter.split_text(text)

from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceBgeEmbeddings

docs = [Document(page_content=chunk)for chunk in text_chunks]

# create embeddings model
embeddings = HuggingFaceBgeEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# create FAISS vector store from documents
vectordb = FAISS.from_documents(docs, embeddings)

st.success('FAISS VectorStore created successfully')
retriever = vectordb.as_retriever()



from langchain.chat_models import init_chat_model
model = init_chat_model(model='gemma2-9b-it', model_provider='groq',api_key= groqapi)

from langchain.prompts import PromptTemplate

template = """
You are a helpful assistant.Anawer the question using only the cntext below.
If the answer is not present, just say so. Do not try to make up an answer.

Context:
{context}

Question:
{question}

Helpful Answer:
"""

rag_prompt = PromptTemplate(input_variables=["context","question"],template=template)

user_query = st.text_input("Ask a question about the PDF")

if user_query:
    relevent_docs = retriever.invoke(user_query)
    final_prompt = rag_prompt.format(context= relevent_docs, question= user_query)

    with st.spinner("Generatin answer..."):
        response = model.invoke(final_prompt)

    st.write("### Answer:")
    st.write(response.content)