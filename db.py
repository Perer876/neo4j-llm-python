from langchain_community.graphs import Neo4jGraph
from config import NEO4J_URL, NEO4J_USERNAME, NEO4J_PASSWORD

graph = Neo4jGraph(
    url=NEO4J_URL,
    username=NEO4J_USERNAME,
    password=NEO4J_PASSWORD,
)

if __name__ == '__main__':
    result = graph.query("""
    MATCH (m:Movie{title: 'Toy Story'}) 
    RETURN m.title, m.plot, m.poster
    """)

    print(result)
