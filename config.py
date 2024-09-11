from dotenv import dotenv_values

env_vars = dotenv_values(".env")

OPENAI_API_KEY = env_vars.get("OPENAI_API_KEY")

if OPENAI_API_KEY is None:
    raise ValueError("OPENAI_API_KEY is not set")

NEO4J_URL = env_vars.get("NEO4J_URL")

if NEO4J_URL is None:
    raise ValueError("NEO4J_URL is not set")

NEO4J_USERNAME = env_vars.get("NEO4J_USERNAME")

if NEO4J_USERNAME is None:
    raise ValueError("NEO4J_USERNAME is not set")

NEO4J_PASSWORD = env_vars.get("NEO4J_PASSWORD")
