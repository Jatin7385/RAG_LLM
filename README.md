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
