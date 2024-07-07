from langchain_community.embeddings.ollama import OllamaEmbeddings
import argparse
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate

from langchain_community.llms.ollama import Ollama


PROMPT_TEMPLATE = """
Answer the question based only on the following context:
{context}

---
Answer the question based on the above context : {question}
"""

CHROMA_PATH = "chroma"

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



# Query

def query_rag(query_text):
    # Prepeare the DB.
    embedder = get_embedding_function(embedder="ollama")

    db = Chroma(persist_directory=CHROMA_PATH, embedding_function = embedder)

    # Search the db
    results = db.similarity_search_with_score(query_text, k = 5)

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)

    model = Ollama(model="mistral")
    response_text = model.invoke(prompt)

    sources = [doc.metadata.get("id", None) for doc, _score in results]
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    print(formatted_response)
    return response_text


def main():
    # Create CLI
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--query_text", type=str, help="Your query...")
    # parser.add_argument("data_path", type=str, help="Data path...")
    # args = parser.parse_args()
    # query_text = args.query_text
    # path = args.data_path

    while True:
        query_text = input("Enter your query here : ")
        # print("Query : ", query_text)
        # query_text = "How to get started with AEM?"
        query_rag(query_text)


if __name__ == "__main__":
    main()