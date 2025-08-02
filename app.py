import streamlit as st
from main import naive_generation, rag_retrieval, rag_generation, KNOWLEDGE_BASE

st.title("RAG Demo: Project Chimera")

query = st.text_input("Enter your query:")

if query:
    st.subheader("Naive LLM Response")
    naive_answer = naive_generation(query)
    st.write(naive_answer)

    st.subheader("RAG Retrieval")
    retrieved_docs = rag_retrieval(query, KNOWLEDGE_BASE)
    st.write(f"Retrieved {len(retrieved_docs)} relevant document(s).")
    for doc in retrieved_docs:
        st.markdown(f"**{doc['title']}**")
        st.write(doc['content'])

    st.subheader("RAG LLM Response")
    rag_answer = rag_generation(query, retrieved_docs)
    st.write(rag_answer)