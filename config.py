from dotenv import dotenv_values

env_vars = dotenv_values(".env")

OPENAI_API_KEY = env_vars.get("OPENAI_API_KEY")

if OPENAI_API_KEY is None:
    raise ValueError("OPENAI_API_KEY is not set")
