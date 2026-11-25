# Ontology and RDF (what your mentor asked about)

## What is an ontology?
A formal vocabulary for a domain: which **classes** exist (User, Post, Hashtag), which **relations**
connect them (posted, retweeted), which **properties** they have, and rules (every Post has exactly
one author). Our ontology is the schema in [GRAPH_SCHEMA.md](GRAPH_SCHEMA.md) plus the label taxonomy.

## What is RDF?
RDF (Resource Description Framework, a W3C standard) stores knowledge as **triples**:
```
subject      predicate      object
ex:u_0042    ex:posted      ex:p_000317 .
ex:p_000317  ex:hasLabel    ex:HATE .
```
Everything is a triple; identifiers are IRIs (web-style names). Queried with **SPARQL**; vocabularies
can be described with RDFS/OWL.

## RDF vs Neo4j property graph
| | RDF | Property graph (Neo4j) |
|---|---|---|
| Unit | triple | node / relationship with properties |
| Properties on edges | needs reification / RDF-star | native (`RETWEETED {timestamp}`) |
| Query language | SPARQL | Cypher |
| Strength | interoperability, linked data, reasoning (OWL) | traversal speed, analytics (GDS), developer ergonomics |

Our timestamped, provenance-carrying interactions are easier as property-graph relationships, and
GDS requires Neo4j, so **RDF serialisation is not needed for the implementation**. The concepts map
directly, so you can describe the schema as an ontology in the report.

## Mapping
| Property graph | RDF |
|---|---|
| `(:User {user_id})` | `ex:u_x rdf:type ex:User` |
| `-[:POSTED]->` | `ex:posted` predicate |
| `Post.timestamp` | `ex:p_x ex:timestamp "2025-03-10T10:00:00Z"^^xsd:dateTime` |
| relationship property (`RETWEETED.timestamp`) | an `ex:Retweet` event node, or RDF-star |

## Optional Turtle example
```turtle
@prefix ex: <http://example.org/gelm#> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .

ex:User a rdfs:Class .  ex:Post a rdfs:Class .  ex:ContentLabel a rdfs:Class .
ex:posted rdfs:domain ex:User ; rdfs:range ex:Post .
ex:hasLabel rdfs:domain ex:Post ; rdfs:range ex:ContentLabel .

ex:u_0042 a ex:User ; ex:synthetic true .
ex:p_000317 a ex:Post ; ex:timestamp "2025-03-10T10:00:00Z"^^xsd:dateTime ; ex:hasLabel ex:HATE .
ex:u_0042 ex:posted ex:p_000317 .
ex:rt_123 a ex:Retweet ; ex:actor ex:u_0007 ; ex:object ex:p_000317 ;
          ex:timestamp "2025-03-10T12:00:00Z"^^xsd:dateTime .   # event-node pattern for edge properties
```
If an RDF export is later required, the Neo4j **neosemantics (n10s)** plugin can export RDF
**[verify compatibility with your Neo4j version]**; no separate RDF database is needed.
