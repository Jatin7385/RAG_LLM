# Data Loader LangChain
from langchain_community.document_loaders import PyPDFDirectoryLoader

# Data Splitter Langchain
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document

# Creating Embeddings
from langchain_community.embeddings.bedrock import BedrockEmbeddings # AWS Cloud
from langchain_community.embeddings.ollama import OllamaEmbeddings # Run completely locally
# import ollama

# Creating the Vector database
from langchain_community.vectorstores import Chroma

# Document Loaders
def load_documents(DATA_PATH):
    document_loader = PyPDFDirectoryLoader(DATA_PATH)
    return document_loader.load()

# Documents are too large so we split them recursively
def split_documents(documents : list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 800,
        chunk_overlap = 80,
        length_function = len,
        is_separator_regex = False,
    )

    return text_splitter.split_documents(documents)

# Embedding Functions - Bedrock is AWS's Embedding Functions. Can switch to another one if required
def get_embedding_function(embedder):

    if(embedder == "Bedrock"):
        embeddings = BedrockEmbeddings(
            credentials_profile_name = "default",
            region_name = "us-east-1"
        )
    elif(embedder == "ollama"):
        embeddings = OllamaEmbeddings(model="nomic-embed-text")

    return embeddings


# Create the vector datastore
def add_to_chroma(chunks, embedder):
    CHROMA_PATH = "chroma"
    # Load the existing database.
    db = Chroma(
        persist_directory=CHROMA_PATH, embedding_function=embedder
    )

    # Calculate Page IDs.
    chunks_with_ids = calculate_chunk_ids(chunks)

    # Add or Update the documents.
    existing_items = db.get(include=[])  # IDs are always included by default
    existing_ids = set(existing_items["ids"])
    print(f"Number of existing documents in DB: {len(existing_ids)}")

    # Only add documents that don't exist in the DB.
    new_chunks = []
    for chunk in chunks_with_ids:
        if chunk.metadata["id"] not in existing_ids:
            new_chunks.append(chunk)

    if len(new_chunks):
        print(f"👉 Adding new documents: {len(new_chunks)}")
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        db.add_documents(new_chunks, ids=new_chunk_ids)
        db.persist()
    else:
        print("✅ No new documents to add")

# Create Chunk IDs for the database : "data/monopoly.pdf:6:2"
# Page Source : Page Number : Chunk Index
def calculate_chunk_ids(chunks):

    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"

        # If the page ID is the same as the last one, increment the index.
        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        # Calculate the chunk ID.
        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id

        # Add it to the page meta-data.
        chunk.metadata["id"] = chunk_id

    return chunks

# DRIVER CODE

# Load the document
# print("Loading data from Path : ", path)
docs = load_documents()

# Split Documents to chunks
chunks = split_documents(docs)

# Get the embedder used to embed the documents
embedder = get_embedding_function("ollama")

# Create the vector data store
add_to_chroma(chunks, embedder)
