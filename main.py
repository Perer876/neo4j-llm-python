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
        ( "system", "When you do not know a beach condition, you should apologize" ),
        ( "system", "{context}" ),
        ( "human", "{question}" ),
    ]
)

chat_chain = prompt | chat_llm | StrOutputParser()

current_weather = """
{
    "surf": [
        {"beach": "Fistral", "conditions": "6ft waves and offshore winds"},
        {"beach": "Polzeath", "conditions": "Flat and calm"},
        {"beach": "Watergate Bay", "conditions": "3ft waves and onshore winds"}
    ]
}"""

if __name__ == '__main__':
    response = chat_chain.invoke({
        "context": current_weather,
        "question": "What is the weather like on Puerto Vallarta?",
    })

    print(response)
