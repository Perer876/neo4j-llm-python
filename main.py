from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
import config

chat_llm = ChatOpenAI(
    openai_api_key=config.OPENAI_API_KEY,
)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a surfer dude, having a conversation about the surf conditions on the beach. Respond using surfer slang.",
        ),
        (
            "human",
            "{question}"
        ),
    ]
)

chat_chain = prompt | chat_llm | StrOutputParser()

if __name__ == '__main__':
    response = chat_chain.invoke({"question": "What's the surf like today?"})

    print(response)
