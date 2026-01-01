import os
import streamlit as st

from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

# ------------------ PAGE CONFIG ------------------
st.set_page_config(page_title="LIC Smart Insurance Assistant", layout="wide")
st.title("🤖 LIC Smart Insurance Assistant (GenAI Powered)")

# ------------------ API KEY ------------------
os.environ["GROQ_API_KEY"] = "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"


# ------------------ MOCK POLICY DATABASE ------------------
policy_database = {
    "LIC12345": {
        "holder": "Rahul Sharma",
        "premium_amount": "₹12,000",
        "due_date": "15-Apr-2026",
        "status": "Active"
    },
    "LIC67890": {
        "holder": "Anita Verma",
        "premium_amount": "₹8,500",
        "due_date": "01-Mar-2026",
        "status": "Active"
    }
}

# ------------------ SIDEBAR ------------------
st.sidebar.header("📄 Upload LIC Policy PDF")
uploaded_file = st.sidebar.file_uploader("Upload LIC Policy Document (PDF)", type="pdf")

# ------------------ LOAD & PROCESS PDF ------------------
@st.cache_resource
def load_vectorstore(pdf_path):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    docs = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    return FAISS.from_documents(docs, embeddings)

# ------------------ INITIALIZE QA CHAIN ------------------
if uploaded_file:
    with open("Policy-Document_LIC-s_New-Jeevan_Amar", "wb") as f:
        f.write(uploaded_file.getbuffer())

    vectorstore = load_vectorstore("Policy-Document_LIC-s_New-Jeevan_Amar.pdf")

    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.2
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever()
    )

    st.success("✅ LIC Policy Document Loaded Successfully")

# ------------------ CHAT INTERFACE ------------------
st.subheader("💬 Ask Your Question")

user_input = st.text_input("Enter your question or policy number:")

if st.button("Ask"):
    if not uploaded_file:
        st.warning("Please upload a LIC policy PDF first.")
    elif user_input.upper() in policy_database:
        policy = policy_database[user_input.upper()]

        st.markdown("### 📄 Policy Details")
        st.write("**Policy Holder:**", policy["holder"])
        st.write("**Premium Amount:**", policy["premium_amount"])
        st.write("**Next Due Date:**", policy["due_date"])
        st.write("**Policy Status:**", policy["status"])

    else:
        response = qa_chain.invoke({"query": user_input})
        st.markdown("### 🤖 Assistant Response")
        st.write(response["result"])

# ------------------ FOOTER ------------------
st.markdown("---")
st.caption("🔐 This is a demo academic project. Real deployment requires LIC backend integration.")
