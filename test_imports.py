import traceback
try:
    from langchain_huggingface import HuggingFaceEmbeddings
    print("HuggingFaceEmbeddings OK")
except Exception as e:
    print("HuggingFaceEmbeddings FAILED")
    traceback.print_exc()

try:
    from langchain_community.vectorstores import FAISS
    print("FAISS OK")
except Exception as e:
    print("FAISS FAILED")
    traceback.print_exc()

try:
    from langchain_groq import ChatGroq
    print("ChatGroq OK")
except Exception as e:
    print("ChatGroq FAILED")
    traceback.print_exc()
