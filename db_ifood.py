from langchain_community.document_loaders import CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings

data_path = "CSV_path"

def generate_db():
    data = load_csv(data_path) # Load csv
    chunks = create_chunks(data) # Splite the document in chunks
    vectorization(chunks) # Embedding Process

def load_csv(data_path):
    loader = CSVLoader(data_path)
    return loader.load()

def create_chunks(data):
    documents_spliter = RecursiveCharacterTextSplitter(chunk_size = 260, chunk_overlap = 65, length_function = len,add_start_index = True)
    chunks = documents_spliter.split_documents(data)
    return chunks

def vectorization(chunks):
    Chroma.from_documents(chunks, OllamaEmbeddings(model="nomic-embed-text"), persist_directory = "database")
    print('Database created!')

generate_db()