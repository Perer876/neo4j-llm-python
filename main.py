import config
from langchain_openai import OpenAI
from templates import ThePrimeagen

llm = OpenAI(
    openai_api_key=config.OPENAI_API_KEY,
    model="gpt-3.5-turbo-instruct",
    temperature=0
)

if __name__ == '__main__':
    response = llm.invoke(ThePrimeagen.format(advice="Which programming language should I learn?"))

    print(response)
