from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
import config

chat_llm = ChatOpenAI(
    openai_api_key=config.OPENAI_API_KEY,
)

instructions = SystemMessage(content="""
You are a surfer dude, having a conversation about the surf conditions on the beach.
Respond using surfer slang.
""")

question = HumanMessage(content="What is the weather like?")

if __name__ == '__main__':
    response = chat_llm.invoke([
        instructions,
        question,
    ])

    print(response.content)
