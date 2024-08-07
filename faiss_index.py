import pickle

from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()
from langchain_community.document_loaders import UnstructuredURLLoader

# Function to fetch text from a URL
def fetch_documents_from_url(urls):
    loader = UnstructuredURLLoader(urls=urls)
    documents = loader.load()
    return documents

# Function to split text into chunks using RecursiveTextSplitter
def split_documents(documents, chunk_size=200, separators=["\n\n", "\n", "."]):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, separators=separators)
    # chunks = []
    # for doc in documents:
    #     chunks.extend(splitter.split_text(doc.page_content))
    # return chunks, splitter.split_documents(documents)
    return splitter.split_documents(documents)

# Function to create FAISS index and save as PKL
def create_vector_store(docs, filepath, embeddings):
    vector_store = FAISS.from_documents(docs, embeddings)
    # Persist the vectors locally on disk
    vector_store.save_local(filepath)
    # return vector_store
    # with open('faiss_index.pkl', 'wb') as f:
    #     pickle.dump(vector_store, f)
    # # print(vector_store)
    # index_components = {
    #     'd': faiss_index.d,  # Dimension of the vectors
    #     'ntotal': faiss_index.ntotal,  # Number of vectors indexed
    #     'vectors': faiss.vector_to_array(faiss_index.reconstruct_n(0, faiss_index.ntotal))  # Extract vectors
    # }

    # Serialize with pickle

    # print(vector_store.search('Provide variety of mechanisms for connecting to Snowflake and executing database commands?'))
    # Use OpenAIEmbeddings to get embeddings
    # embeddings = OpenAIEmbeddings()

    # Create FAISS index
    # faiss_index = FAISS.from_documents(docs, embeddings)
    # vector_store = FAISS.from_documents(docs, embeddings)

    # # Load pre-trained model and tokenizer from transformers
    # tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    # model = AutoModel.from_pretrained("distilbert-base-uncased")
    #
    # # Tokenize and embed texts
    # embeddings = []
    # for chunk in chunks:
    #     inputs = tokenizer(chunk, return_tensors='pt', truncation=True, padding=True, max_length=512)
    #     with torch.no_grad():
    #         outputs = model(**inputs)
    #     embeddings.append(outputs.last_hidden_state.mean(dim=1).squeeze().numpy())
    #
    # print(embeddings)
    # # Create FAISS index
    # faiss_index = FAISS(embedding_size=embeddings[0].shape[0])
    # faiss_index.add_items(embeddings)

    # Load pre-trained SentenceTransformer model
    # model = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')
    # model = SentenceTransformer('all-MiniLM-L6-v2')

    # Embed texts
    # embeddings = model.encode(chunks)
    # print(embeddings)


    # Convert embeddings to a numpy array of type float32 for FAISS
    # import numpy as np
    # embeddings_np = np.array(embeddings).astype('float32')

    # Create a FAISS index
    # dimension = embeddings_np.shape[1]  # Dimension of the embeddings
    # print(len(embeddings_np), len(embeddings_np[0]))
    # print(embeddings_np[0])
    # faiss_index = faiss.IndexFlatL2(dimension)  # Using L2 distance for similarity search
    # faiss_index.add(embeddings_np)
    # with open('faiss_index.pkl', 'wb') as f:
    #     pickle.dump(faiss_index, f)

urls = ["https://www.livemint.com/market/stock-market-news/wall-street-today-us-stocks-rise-after-global-markets-rout-11722950046752.html"]
# Example URLs (replace with actual URLs)
if __name__=='__main__':
    pass



