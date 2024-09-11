import config

from langchain_openai import OpenAI

llm = OpenAI(
    openai_api_key=config.OPENAI_API_KEY,
)

if __name__ == '__main__':
    response = llm.invoke("What is Neo4j?")

    print(response)
