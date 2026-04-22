import os
from dotenv import load_dotenv
from neo4j import GraphDatabase
from openai import OpenAI
from typing import List, Dict, Any

load_dotenv()


class Neo4jQueryInterface:
    
    def __init__(self, uri: str = None, username: str = None, password: str = None):
        self.uri = uri or os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.username = username or os.getenv("NEO4J_USERNAME", "neo4j")
        self.password = password or os.getenv("NEO4J_PASSWORD", "password")
        
        self.driver = GraphDatabase.driver(self.uri, auth=(self.username, self.password))
        
        api_key = os.getenv("OPENAI_API_KEY")
        self.openai_client = OpenAI(api_key=api_key)
    
    def close(self):
        self.driver.close()
    
    def get_database_schema(self) -> str:
        with self.driver.session() as session:
            node_result = session.run("CALL db.labels()")
            labels = [record[0] for record in node_result]
            
            rel_result = session.run("CALL db.relationshipTypes()")
            relationships = [record[0] for record in rel_result]
            
            prop_result = session.run("CALL db.propertyKeys()")
            properties = [record[0] for record in prop_result]
            
            schema = f"""
                Database Schema:
                - Node Labels: {', '.join(labels)}
                - Relationship Types: {', '.join(relationships)}
                - Property Keys: {', '.join(properties)}
                """
            return schema.strip()
    
    def translate_to_cypher(self, natural_language_query: str, schema: str) -> str:
        system_prompt = f"""You are a Neo4j Cypher query expert specializing in knowledge graphs.
Based on the following database schema, translate the user's natural language query into a Cypher query statement.

{schema}

Requirements:
1. Return only the Cypher query statement without any explanation
2. Ensure correct syntax
3. Use standard Cypher syntax like MATCH, WHERE, RETURN
4. IMPORTANT: When querying for information about an entity, find ALL related entities but AVOID DUPLICATE RESULTS
5. Use relationship traversal patterns to discover connected nodes:
   - Prefer returning distinct unique entities instead of paths
   - Use distinct n, rel, m to avoid duplicate result rows
   - For multi-level relationships, use UNION to combine results or COLLECT for aggregation
6. Example patterns to follow (use DISTINCT to avoid duplicates):
   - Direct relationships: MATCH (n {{name: 'Entity'}})-[rel]-(m) RETURN distinct n, rel, m
   - Multi-level with UNION: MATCH (n {{name: 'Entity'}})-[rel]-(m) RETURN n, rel, m UNION MATCH (n {{name: 'Entity'}})-[*2]-(m) RETURN distinct n, rel, m
   - Aggregated: MATCH (n {{name: 'Entity'}})-[rel]-(m) RETURN n, collect(distinct {{entity: m, relationship: type(rel)}}) as related_entities
7. Avoid returning all paths - instead return unique nodes and their direct relationships
8. Always include DISTINCT or LIMIT to control result size and prevent duplicates
"""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": natural_language_query}
        ]
        
        response = self.openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.3
        )
        
        cypher_query = response.choices[0].message.content.strip()
        cypher_query = cypher_query.replace("```cypher", "").replace("```", "").strip()
        
        return cypher_query
    
    def execute_cypher(self, cypher_query: str) -> List[Dict[str, Any]]:
        with self.driver.session() as session:
            result = session.run(cypher_query)
            records = [record.data() for record in result]
            return records
    
    def results_to_natural_language(self, 
                                     query: str, 
                                     cypher: str, 
                                     results: List[Dict[str, Any]]) -> str:
        system_prompt = """You are a data analysis assistant.
Convert database query results into clear and easy-to-understand natural language answers.
Be concise, accurate, and directly answer the user's question.
"""
        
        user_message = f"""
User Question: {query}

Executed Query: {cypher}

Query Results:
{results}

Please summarize these results in natural language to answer the user's question.
"""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]
        
        response = self.openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.5
        )
        
        return response.choices[0].message.content.strip()
    
    def query(self, natural_language_query: str) -> Dict[str, Any]:
        try:
            print("📊 Fetching database schema...")
            schema = self.get_database_schema()
            
            print("🔄 Translating to Cypher...")
            cypher_query = self.translate_to_cypher(natural_language_query, schema)
            print(f"Generated Cypher: {cypher_query}")
            
            print("⚡ Executing query...")
            results = self.execute_cypher(cypher_query)
            
            print("💬 Generating natural language response...")
            natural_response = self.results_to_natural_language(
                natural_language_query, 
                cypher_query, 
                results
            )
            
            return {
                "query": natural_language_query,
                "schema": schema,
                "cypher": cypher_query,
                "results": results,
                "response": natural_response
            }
        
        except Exception as e:
            return {
                "query": natural_language_query,
                "error": str(e),
                "response": f"Query failed: {str(e)}"
            }


if __name__ == "__main__":
    interface = Neo4jQueryInterface()

    query = "tell me something about Google"
    result = interface.query(query)
    print(f"Cypher: {result.get('cypher')}")
    print(f"Raw results: {result.get('results')}")
    print("\n")
    print(result["response"])

    interface.close()
