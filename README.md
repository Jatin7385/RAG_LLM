# Retrieval Augmented Generation

Playing around with RAG - LLM's exploring certain use cases. 
- Open Source LLM model used : Mistral
- Vector Data Store used : Chroma DB
- Embeddings Function : Ollama Nomic-Embed-Text
- Data used : AEM Guide

## To Setup the project : 
1) pip install langchain-community
2) pip install chromadb(If build fails for chromadb-hnswlib, you need to sudo apt-get install libfuse I think it was. lib something something atlease)
3) Setup Ollama -> curl -fsSL https://ollama.com/install.sh | sh
4) ollama serve
5) ollama pull nomic-embed-text
6) ollama pull mistral

## To Run the project : 
1) Change the path of data in data_loader.py/load_documents()(DATA_PATH)
2) Run data_loader.py
3) Run query_data.py
4) Enter your queries, get your replies

## Test Run of the Project
<img width="960" alt="image" src="https://github.com/Jatin7385/RAG_LLM/assets/73430464/b2eaff12-5bd4-4819-8250-0fc0679528a9">
