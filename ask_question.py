
import pickle

import langchain
import numpy as np
from dotenv import load_dotenv
from langchain.chains.qa_with_sources.retrieval import RetrievalQAWithSourcesChain
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from transformers import pipeline
load_dotenv()
from faiss_index import urls, fetch_documents_from_url, split_documents


# Load FAISS index from PKL
def load_faiss_index(embeddings, filename='faiss_index_constitution'):
    # Load from local storage
    persisted_vectorstore = FAISS.load_local(filename, embeddings, allow_dangerous_deserialization=True)
    return persisted_vectorstore


# Embed the query using the same model
def embed_query(query, model):
    return model.encode([query])[0]


# Perform the search and get relevant chunks
def search_faiss_index(query, faiss_index, model, top_k=5):
    query_embedding = embed_query(query, model)
    query_embedding = np.expand_dims(query_embedding, axis=0).astype(np.float32)

    distances, indices = faiss_index.search(query_embedding, top_k)
    return distances, indices


# Example function to retrieve and process results
def retrieve_and_process_results(query, faiss_index, model, chunks):
    distances, indices = search_faiss_index(query, faiss_index, model)

    # Retrieve the relevant chunks based on indices
    relevant_chunks = [chunks[i] for i in indices[0]]

    # Here, you could further process these chunks or use them to generate an answer
    return relevant_chunks


def filter_results_with_transformers(query, relevant_chunks):
    # Prepare prompt for the model
    context = ''.join(relevant_chunks)
    prompt = f"Given the following context, answer the question:\n\nQuestion: {query}\n\nContext: {context}"

    # Use the text generation pipeline
    generator = pipeline('text-generation', model='bert-base-uncased')  # Replace with a suitable model
    # generator = pipeline('text-generation', model='Meta-Llama-3.1-405B')  # Replace with a suitable model

    # Generate response
    response = generator(prompt, max_length=450, num_return_sequences=1)
    print('response', response)
    return response[0]['generated_text'].strip()

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
# Example query
def main(query):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = load_faiss_index(embeddings)
    chain = RetrievalQAWithSourcesChain.from_llm(llm= llm, retriever=vector_store.as_retriever())
    # print(chain)
    # langchain.debug = True
    result = chain({'question':query}, return_only_outputs=True)
    print(result)

# Example query (replace with your actual query)
# query = "Information about Oil prices in global markets"
# main(query)
