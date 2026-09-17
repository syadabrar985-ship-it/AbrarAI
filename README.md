Ask Abrar — Personal RAG Chatbot
A Streamlit RAG application that answers questions about Syed Abrar Hussain Shah.

Stack
Python

Streamlit

Sentence Transformers (all-MiniLM-L6-v2) for open-source embeddings

FAISS for vector search

Groq API

openai/gpt-oss-120b for generation

Files
app.py — application

knowledge_base.txt — personal profile used as the source of truth

requirements.txt — Python dependencies

Run locally
pip install -r requirements.txt
streamlit run app.py
Set your Groq key as GROQ_API_KEY, or put it in Streamlit Secrets.

Streamlit Community Cloud
Push these files to GitHub.

Create a new Streamlit app from the repository.

Select app.py as the main file.

Add this secret:

GROQ_API_KEY = "your-groq-api-key"
Do not commit your API key to GitHub.

Retrieval improvements
This version uses:

section-aware chunking

semantic FAISS retrieval

lexical keyword retrieval

typo correction

heading-aware ranking

conversational question handling

a strict source-of-truth prompt

So questions such as what is your name?, what is your education?, and common misspellings such as eduction can retrieve the relevant profile section instead of immediately falling back to a generic "I don't have that information" response.

