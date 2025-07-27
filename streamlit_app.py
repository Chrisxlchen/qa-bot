import streamlit as st
import requests
import os
from dotenv import load_dotenv

load_dotenv()

API_BASE_URL = "http://localhost:8000"

st.set_page_config(
    page_title="QA Bot",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Document Q&A Bot")
st.markdown("Ask questions about your documents and get AI-powered answers!")

with st.sidebar:
    st.header("Configuration")
    
    if st.button("📚 Index Documents"):
        with st.spinner("Indexing documents..."):
            try:
                response = requests.post(f"{API_BASE_URL}/index", json={"force_reindex": False})
                if response.status_code == 200:
                    st.success("Documents indexed successfully!")
                    st.json(response.json())
                else:
                    st.error(f"Error: {response.text}")
            except requests.exceptions.ConnectionError:
                st.error("Cannot connect to API. Make sure the FastAPI server is running on port 8000.")
            except Exception as e:
                st.error(f"Error: {str(e)}")
    
    if st.button("🔄 Re-index Documents"):
        with st.spinner("Re-indexing documents..."):
            try:
                response = requests.post(f"{API_BASE_URL}/index", json={"force_reindex": True})
                if response.status_code == 200:
                    st.success("Documents re-indexed successfully!")
                    st.json(response.json())
                else:
                    st.error(f"Error: {response.text}")
            except requests.exceptions.ConnectionError:
                st.error("Cannot connect to API. Make sure the FastAPI server is running on port 8000.")
            except Exception as e:
                st.error(f"Error: {str(e)}")
    
    if st.button("📊 Show Stats"):
        try:
            response = requests.get(f"{API_BASE_URL}/stats")
            if response.status_code == 200:
                stats = response.json()
                st.json(stats)
            else:
                st.error(f"Error: {response.text}")
        except requests.exceptions.ConnectionError:
            st.error("Cannot connect to API. Make sure the FastAPI server is running on port 8000.")
        except Exception as e:
            st.error(f"Error: {str(e)}")

col1, col2 = st.columns([2, 1])

with col1:
    question = st.text_input("💭 Ask a question about your documents:", 
                           placeholder="e.g., What is the main topic discussed in the documents?")

with col2:
    n_results = st.slider("Number of relevant chunks to retrieve:", 1, 10, 5)

if st.button("🔍 Ask Question", type="primary"):
    if question:
        with st.spinner("Searching for answers..."):
            try:
                response = requests.post(f"{API_BASE_URL}/ask", 
                                       json={"question": question, "n_results": n_results})
                
                if response.status_code == 200:
                    result = response.json()
                    
                    if "error" in result:
                        st.error(result["error"])
                    else:
                        st.markdown("### 🎯 Answer")
                        st.markdown(result["answer"])
                        
                        if result.get("sources"):
                            st.markdown("### 📚 Sources")
                            for i, source in enumerate(result["sources"], 1):
                                st.markdown(f"{i}. `{source}`")
                        
                        st.markdown(f"**Context used:** {result.get('context_used', 0)} document chunks")
                else:
                    st.error(f"Error: {response.text}")
                    
            except requests.exceptions.ConnectionError:
                st.error("Cannot connect to API. Make sure the FastAPI server is running on port 8000.")
            except Exception as e:
                st.error(f"Error: {str(e)}")
    else:
        st.warning("Please enter a question.")

st.markdown("---")
st.markdown("### 📖 How to use:")
st.markdown("""
1. **Place your documents** in the `documents/` folder (supports PDF, TXT, MD files)
2. **Index documents** using the sidebar button
3. **Ask questions** about your documents
4. **Get AI-powered answers** based on the document content
""")

st.markdown("### ⚙️ Setup:")
st.markdown("""
1. Copy `.env.example` to `.env` and add your OpenAI API key
2. Run the FastAPI server: `python app.py`
3. Run this Streamlit app: `streamlit run streamlit_app.py`
""")