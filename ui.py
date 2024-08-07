import time

import streamlit as st
from dotenv import load_dotenv
from langchain.chains.qa_with_sources.retrieval import RetrievalQAWithSourcesChain
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

from ask_question import load_faiss_index
from faiss_index import create_vector_store, fetch_documents_from_url, split_documents
load_dotenv()
st.title("Bot: News Research Tool 📈")
st.sidebar.title("News Article URLs")

urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")
file_path = "faiss_index_constitution"

main_placeholder = st.empty()
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
if process_url_clicked:
    main_placeholder.text("Data Loading...Started...✅✅✅")
    documents = fetch_documents_from_url(urls)
    main_placeholder.text("Text Splitter...Started...✅✅✅")
    docs = split_documents(documents)
    main_placeholder.text("Embedding Vector Started Building...✅✅✅")
    time.sleep(2)
    create_vector_store(docs, file_path, embeddings)

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
query = main_placeholder.text_input("Question: ")
if query:
    vector_store = load_faiss_index(embeddings, file_path)
    chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vector_store.as_retriever())
    result = chain({'question': query}, return_only_outputs=True)
    st.header("Answer")
    st.write(result["answer"])
    # Display sources, if available
    sources = result.get("sources", "")
    if sources:
        st.subheader("Sources:")
        sources_list = sources.split("\n")  # Split the sources by newline
        for source in sources_list:
            st.write(source)

