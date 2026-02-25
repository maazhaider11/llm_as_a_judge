from neo4j import GraphDatabase

uri = "bolt://neo4j:7687"
user = "neo4j"
password = "password"

driver = GraphDatabase.driver(uri, auth=(user, password))

dummy_triples = [
    {"subject": "Earth", "predicate": "orbits", "object": "Sun"},
    {"subject": "Moon", "predicate": "orbits", "object": "Earth"},
    {"subject": "Water", "predicate": "has_boiling_point", "object": "100_Celsius"},
]

def add_data():
    with driver.session() as session:
        for triple in dummy_triples:
            session.run(
                "MERGE (s:Entity {name: $sub}) MERGE (o:Entity {name: $obj}) MERGE (s)-[r:RELATED_TO {predicate: $pred}]->(o)",
                sub=triple["subject"], obj=triple["object"], pred=triple["predicate"]
            )
        print("Successfully added dummy data to Neo4j.")

if __name__ == "__main__":
    add_data()
