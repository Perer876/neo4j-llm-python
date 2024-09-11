from langchain_core.output_parsers import SimpleJsonOutputParser
from langchain_openai import OpenAI
from templates import template
import config

llm = OpenAI(
    openai_api_key=config.OPENAI_API_KEY,
    model="gpt-3.5-turbo-instruct",
    temperature=0.1
)

llm_chain = template | llm | SimpleJsonOutputParser()

if __name__ == '__main__':
    response = llm_chain.invoke({
        "fruit": 'Orange',
    })

    print(response)
