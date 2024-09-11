import config
from langchain_openai import OpenAI
from templates import ThePrimeagen

llm = OpenAI(
    openai_api_key=config.OPENAI_API_KEY,
)

if __name__ == '__main__':
    response = llm.invoke(ThePrimeagen.format(advice="how to be a good at swimming"))

    print(response)
