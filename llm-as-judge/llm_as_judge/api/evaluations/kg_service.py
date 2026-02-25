from neo4j import Session
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass
import re

@dataclass
class Triple:
    """Representation of a knowledge graph triple"""
    subject: str
    predicate: str
    object: str
    source: Optional[str] = None

class KGService:
    """
    Service to handle Knowledge Graph operations in Neo4j.
    """
    def __init__(self, neo4j_session: Session):
        self.session = neo4j_session

    def _sanitize_relationship_type(self, predicate: str) -> str:
        sanitized = re.sub(r'[^a-zA-Z0-9_]', '_', predicate).upper()
        sanitized = re.sub(r'_{2,}', '_', sanitized)
        return sanitized.strip('_') or "RELATED_TO"

    def ingest_triples(self, triples: List[Triple], document_id: str) -> Tuple[int, int]:
        nodes_created = 0
        relationships_created = 0

        # Create Document node
        self.session.run(
            "MERGE (d:Document {id: $id}) ON CREATE SET d.created_at = datetime()",
            id=document_id
        )

        for triple in triples:
            rel_type = self._sanitize_relationship_type(triple.predicate)
            
            # Use a single query to merge nodes and relationship for efficiency
            query = f"""
            MERGE (s:Entity {{name: $subject}})
            ON CREATE SET s.created_at = datetime(), s.type = 'Generic'
            MERGE (o:Entity {{name: $object}})
            ON CREATE SET o.created_at = datetime(), o.type = 'Generic'
            MERGE (s)-[r:{rel_type}]->(o)
            ON CREATE SET 
                r.created_at = datetime(),
                r.predicate = $predicate,
                r.source = $source,
                r.document_id = $document_id
            WITH s, o
            MATCH (d:Document {{id: $document_id}})
            MERGE (d)-[:CONTAINS]->(s)
            MERGE (d)-[:CONTAINS]->(o)
            """
            self.session.run(
                query,
                subject=triple.subject,
                object=triple.object,
                predicate=triple.predicate,
                source=triple.source,
                document_id=document_id
            )
            relationships_created += 1

        return nodes_created, relationships_created

    def get_relevant_knowledge(self, query: str, agent_output: str) -> List[Dict[str, Any]]:
        """
        Retrieve triples relevant to the query and agent output from the Knowledge Graph.
        """
        # Simple entity extraction using regex (matches capitalized words/phrases)
        entities = set(re.findall(r'\b[A-Z][a-z]+(?:\s[A-Z][a-z]+)*\b', query + " " + agent_output))
        
        if not entities:
            return []

        results = []
        # Find relationships where subject or object matches extracted entities
        cypher_query = """
        MATCH (s:Entity)-[r]->(o:Entity)
        WHERE s.name IN $entities OR o.name IN $entities
        RETURN s.name as subject, type(r) as predicate, o.name as object, r.predicate as original_predicate
        LIMIT 50
        """
        
        db_results = self.session.run(cypher_query, entities=list(entities))
        for record in db_results:
            results.append({
                "subject": record["subject"],
                "predicate": record["original_predicate"] or record["predicate"],
                "object": record["object"]
            })
        
        return results
