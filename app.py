from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from youtube_transcript_api import YouTubeTranscriptApi , TranscriptsDisabled
from langchain.chains import RetrievalQA
import streamlit as st
import re
import os
from dotenv import load_dotenv
load_dotenv()

st.set_page_config(page_title="Question-Answer Chatbot Based On YouTube Video")
st.title("A RAG-Based Chatbot for YouTube")

groq_api_key = st.sidebar.text_input("Groq_API_Key",type = "password")

if not groq_api_key:
    print("Please provide valid Groq API Key")
    st.stop()

youtube_url = st.text_input("Enter Youtube URL")
#question = st.text_input("Enter your question related to youtube video :")

def extract_video_id(url):
    """Simple Video id extractor"""
    pattern = r"(?:v=|\/)([0-9A-Za-z_-]{11}).*"
    match = re.search(pattern , url)
    video_id = match.group(1)
    return video_id if match else None

def get_transcripts(video_id):
    try:
        transcripts_list = YouTubeTranscriptApi.get_transcript(video_id,languages=["en"])
        transcripts = " ".join((chunk["text"]) for chunk in transcripts_list)
        return transcripts
    except TranscriptsDisabled:
        print("Transcripts are not available")

def create_vectorstore_from_text(text):
    """Split , embed and store vectors to Faiss"""
    splitter = RecursiveCharacterTextSplitter(chunk_size = 1000 , chunk_overlap = 200)
    chunks = splitter.create_documents([text])
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks , embeddings)
    return vectorstore

def build_qa_chain(vectorstore):
    """Creating Retriever QA chain using ChatGroq --> Llama"""
    llm = ChatGroq(groq_api_key = groq_api_key , model = "meta-llama/llama-4-maverick-17b-128e-instruct" , temperature = 0.5)
    retriever = vectorstore.as_retriever()
    qa_chain = RetrievalQA.from_chain_type(llm = llm , retriever = retriever)
    return qa_chain

if st.button("🧠 Load Transcript and Build Chatbot"):
    if not groq_api_key or not youtube_url:
        st.warning("Please provide both Groq_api_key and Video_url")

    else:
        try:
            video_id = extract_video_id(youtube_url)
            with st.spinner("⏳ Fetching transcript..."):
                transcript = get_transcripts(video_id)

            with st.spinner("🔍 Chunking and embedding..."):
                vector_store = create_vectorstore_from_text(transcript)

            qa_chain = build_qa_chain(vector_store)
            st.session_state.qa_chain = qa_chain
            st.success("✅ Chatbot ready! Ask below 👇")

        except Exception as e:
            st.error(f"❌ Error: {str(e)}")

if "qa_chain" in st.session_state:
    question = st.text_input("Enter your question about video")
    if st.button("Ask"):
        if question.strip() != "":
            with st.spinner("Thinking..."):
                answer = st.session_state.qa_chain.run(question)
                st.markdown(f"Answer : {answer}")
