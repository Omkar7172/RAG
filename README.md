(Retrieval-Augmented Generation) Streamlit app:

---

# 🧠 RAG App - PDF Q\&A using FAISS, HuggingFace & Groq

This is a simple Retrieval-Augmented Generation (RAG) application built with **Streamlit**, **LangChain**, and **FAISS**. The app reads a PDF (e.g., a resume), indexes its content, and allows users to ask questions. It returns relevant answers based on the content of the PDF using an LLM powered by **Groq**.

---

## 🔧 Features

* Upload and process PDF documents.
* Chunk and embed text using **HuggingFace** embeddings.
* Store and retrieve relevant text using **FAISS** vector store.
* Generate answers using **Groq** LLM via **LangChain**.
* Interactive user interface with **Streamlit**.

---

## 📁 Project Structure

```
rag-app/
│
├── app.py               # Main Streamlit app
├── requirements.txt     # Dependencies list
└── README.md            # Project documentation
```

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/rag-app.git
cd rag-app
```

### 2. Install Requirements

```bash
pip install -r requirements.txt
```

### 3. Set the Groq API Key

In your `app.py`, set your Groq API key:

```python
groqapi = 'YOUR_GROQ_API_KEY'
```

### 4. Run the App

```bash
streamlit run app.py
```

---

## 📌 How It Works

1. **PDF Reading**: Uses `PyPDF2` to extract text from the uploaded PDF.
2. **Text Splitting**: Splits the extracted text into chunks using `RecursiveCharacterTextSplitter`.
3. **Embedding**: Converts chunks into vector representations using `HuggingFace` embeddings (`all-MiniLM-L6-v2`).
4. **Vector Store**: Stores embeddings in an in-memory FAISS vector store for fast similarity search.
5. **Retrieval**: Retrieves relevant chunks based on user input.
6. **Answer Generation**: Constructs a prompt and generates a response using Groq's LLM (`gemma2-9b-it`).

---

## 📦 Dependencies

* `streamlit`
* `PyPDF2`
* `langchain`
* `langchain_community`
* `sentence-transformers`
* `faiss-cpu`
* `huggingface_hub`

You can install all required packages using:

```bash
pip install -r requirements.txt
```

Sample `requirements.txt`:

```txt
streamlit
PyPDF2
langchain
langchain-community
sentence-transformers
faiss-cpu
huggingface_hub
```

---

## 📝 Example Prompt Template

```txt
You are a helpful assistant. Answer the question using only the context below.
If the answer is not present, just say so. Do not try to make up an answer.

Context:
{context}

Question:
{question}

Helpful Answer:
```

---

## ❓ Example Usage

* Upload a PDF (e.g., your resume).
* Type a question like:
  *"What is the candidate’s educational background?"*
* The app returns an answer based on the content in the PDF.

---

## ⚠️ Notes

* Ensure your Groq API key is valid.
* You can modify the PDF path or make it a file uploader (`st.file_uploader`) for more flexibility.
* This app is intended for educational/demo use and may need enhancements for production.

---

## 🧠 Future Improvements

* Add file uploader instead of hardcoded path.
* Support for multiple PDFs.
* Cache vector DB for performance.
* UI enhancements for better UX.

---

## 📄 License

MIT License

---


