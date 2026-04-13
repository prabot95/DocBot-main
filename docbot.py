import os
import tempfile
import streamlit as st

from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PDFPlumberLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

# Constants
DB_FAISS_PATH = "vectorstore/db_faiss"
MAX_FILE_SIZE_MB = 10

@st.cache_resource
def get_embedding_model():
    """Load and cache the embedding model."""
    return HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

@st.cache_resource
def get_encyclopedia_vectorstore():
    """Load the pre-built encyclopedia vector store from disk."""
    embedding_model = get_embedding_model()
    db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return db

def process_uploaded_pdf(uploaded_file):
    """Process an uploaded PDF and create an in-memory FAISS vector store."""
    # Save uploaded file to a temporary location
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name
    
    try:
        # Load the PDF
        loader = PDFPlumberLoader(tmp_path)
        documents = loader.load()
        
        # Split into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
        text_chunks = text_splitter.split_documents(documents)
        
        # Create embeddings and FAISS index
        embedding_model = get_embedding_model()
        vectorstore = FAISS.from_documents(text_chunks, embedding_model)
        
        return vectorstore, len(text_chunks)
    finally:
        # Clean up temp file
        os.unlink(tmp_path)

def set_custom_prompt(custom_prompt_template):
    """Create a prompt template."""
    prompt = PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
    return prompt

def get_qa_chain(vectorstore):
    """Create a RetrievalQA chain with the given vector store."""
    CUSTOM_PROMPT_TEMPLATE = """
        Use the pieces of information provided in the context to answer user's question.
        If you dont know the answer,then search on your parameters and then answer.

        Context: {context}
        Question: {question}

        Start the answer directly. No small talk please.
        """
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=ChatGroq(
            model_name="llama-3.3-70b-versatile",
            temperature=0.0,
            groq_api_key=os.environ["GROQ_API_KEY"],
        ),
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),
        return_source_documents=True,
        chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
    )
    return qa_chain

def main():
    st.set_page_config(page_title="DocBot - PDF reader", page_icon="📚", layout="wide")
    
    # Sidebar for mode selection
    with st.sidebar:
        st.header("Settings")
        
        mode = st.radio(
            "Select Mode",
            ["Encyclopedia", "Your PDF"],
            help="Choose between the built-in encyclopedia or upload your own PDF"
        )
        
        st.divider()
        
        # PDF Upload section (only shown in "Your PDF" mode)
        if mode == "Your PDF":
            st.subheader("Upload PDF")
            uploaded_file = st.file_uploader(
                "Choose a PDF file",
                type=["pdf"],
                help=f"Maximum file size: {MAX_FILE_SIZE_MB} MB"
            )
            
            if uploaded_file is not None:
                # Check file size
                file_size_mb = len(uploaded_file.getvalue()) / (1024 * 1024)
                
                if file_size_mb > MAX_FILE_SIZE_MB:
                    st.error(f"File too large! Maximum size is {MAX_FILE_SIZE_MB} MB. Your file: {file_size_mb:.1f} MB")
                    uploaded_file = None
                else:
                    st.success(f"File: {uploaded_file.name} ({file_size_mb:.1f} MB)")
                    
                    # Process if not already processed or if file changed
                    current_file_name = uploaded_file.name
                    if (st.session_state.get('uploaded_file_name') != current_file_name):
                        with st.spinner("Processing PDF... This may take a moment."):
                            try:
                                vectorstore, num_chunks = process_uploaded_pdf(uploaded_file)
                                st.session_state['user_vectorstore'] = vectorstore
                                st.session_state['uploaded_file_name'] = current_file_name
                                st.session_state['messages'] = []  # Clear chat on new upload
                                st.info(f"Created {num_chunks} text chunks from your PDF")
                            except Exception as e:
                                st.error(f"Error processing PDF: {str(e)}")
                                st.session_state['user_vectorstore'] = None
        else:
            # Clear user vectorstore when switching to Encyclopedia mode
            if 'user_vectorstore' in st.session_state:
                del st.session_state['user_vectorstore']
                del st.session_state['uploaded_file_name']
                st.session_state['messages'] = []  # Clear chat on mode switch
        
        st.divider()
        st.caption("Powered by Llama 3.3 via Groq")
    
    # Main content
    if mode == "Encyclopedia":
        st.title("Medical Encyclopedia Chatbot")
        st.caption("Ask questions about the GALE Encyclopedia of Medicine")
    else:
        st.title("PDF Q&A Chatbot")
        st.caption("Upload a PDF and ask questions about it")
    
    # Initialize chat history
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    # Display chat history
    for message in st.session_state.messages:
        st.chat_message(message['role']).markdown(message['content'])
    
    # Determine if we can accept queries
    can_query = False
    vectorstore = None
    
    if mode == "Encyclopedia":
        try:
            vectorstore = get_encyclopedia_vectorstore()
            can_query = True
        except Exception as e:
            st.error(f"Failed to load encyclopedia: {str(e)}")
    else:
        if 'user_vectorstore' in st.session_state and st.session_state['user_vectorstore'] is not None:
            vectorstore = st.session_state['user_vectorstore']
            can_query = True
        else:
            st.info("Please upload a PDF file using the sidebar to get started.")
    
    # Chat input
    if can_query:
        prompt = st.chat_input("Ask your question here...")
        
        if prompt:
            st.chat_message('user').markdown(prompt)
            st.session_state.messages.append({'role': 'user', 'content': prompt})
            
            try:
                qa_chain = get_qa_chain(vectorstore)
                
                with st.spinner("Thinking..."):
                    response = qa_chain.invoke({'query': prompt})
                
                result = response["result"]
                source_documents = response["source_documents"]
                
                # Extract page numbers for cleaner display
                pages = set(doc.metadata.get('page', 'N/A') for doc in source_documents)
                pages_str = ", ".join(str(p) for p in sorted(pages) if p != 'N/A')
                
                if pages_str:
                    result_to_show = f"{result}\n\n*Sources: Pages {pages_str}*"
                else:
                    result_to_show = result
                
                st.chat_message('assistant').markdown(result_to_show)
                st.session_state.messages.append({'role': 'assistant', 'content': result_to_show})
            
            except Exception as e:
                st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
