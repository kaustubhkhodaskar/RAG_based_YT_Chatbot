# 🎥 YouTube RAG Chatbot

An AI-powered chatbot that answers questions based on the transcript of any YouTube video using Retrieval-Augmented Generation (RAG) and Llama-4-Maverick-17b.

---

## Feature

- 🔎 Extracts Youtube Transcripts Automatically
- 📚 Splits Transcripts Into Meaningfull Chunks
- 🧠 Uses Huggingface Embeddings model and FAISS Vectorstore
- 💬 Answer Questions Using Llama-4
- 🖥️ Easy To Use Streamlit Web Interface

---

## 🛠️ Installation

- git clone https://github.com/yourusername/youtube-rag-chatbot.git
- cd youtube-rag-chatbot
- pip install -r requirements.txt

## Usage 

- 1. Streamlit run app.py
- 2. Enter your Groq API Key and a YouTube video URL
- 3. Ask questions based on the video content!

## Tech Stack

- Python 🐍
- LangChain
- Huggingface 
- Chat_Groq
- FAISS for vector search
- Streamlit for frontend
- YouTube Transcript API
