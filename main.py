from langchain_core.runnables import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import StrOutputParser
from langchain_community.chat_message_histories import ChatMessageHistory
import config

memory = ChatMessageHistory()

def get_memory(session_id):
    return memory

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
        MessagesPlaceholder(variable_name="chat_history"),
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

chat_with_message_history = RunnableWithMessageHistory(
    chat_chain,
    get_memory,
    input_messages_key="question",
    history_messages_key="chat_history",
)

if __name__ == '__main__':
    while True:
        question = input("> ")

        response = chat_with_message_history.invoke(
            {
                "context": current_weather,
                "question": question,

            },
            config={
                "configurable": {"session_id": "none"}
            }
        )

        print(response)
