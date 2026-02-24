# Neo4j GraphRAG ETL Pipeline for WCAG 2.2

## 🎯 Project Overview

This project implements a **Knowledge Graph-based ETL pipeline** for WCAG 2.2 (Web Content Accessibility Guidelines) Success Criteria, designed to power **GraphRAG (Graph Retrieval-Augmented Generation)** applications. The pipeline transforms the WCAG 2.2 specification into a richly interconnected graph database that serves as the foundation for LLM-powered accessibility compliance systems.

### Purpose

- **Extract** WCAG 2.2 specification data from structured JSON
- **Transform** flat data into a semantic knowledge graph with explicit relationships
- **Load** the graph into Neo4j for efficient querying and GraphRAG implementation
- **Enable** LLM training and retrieval-augmented generation for accessibility compliance

---

## 🏗️ Architecture

### Knowledge Graph Structure

The pipeline creates a hierarchical, multi-layered knowledge graph:

```
┌─────────────────────────────────────────────────────────────┐
│                    WCAG Knowledge Graph                      │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  WCAGPrinciple (4 nodes)                                     │
│    ↑ PART_OF                                                 │
│  WCAGGuideline (13 nodes)                                    │
│    ↑ PART_OF                                                 │
│  WCAGCriterion (86+ nodes) ──→ HAS_LEVEL ──→ ConformanceLevel│
│    ↓ HAS_SPECIAL_CASE                                        │
│  WCAGSpecialCase                                             │
│    ↓ HAS_NOTE                                                │
│  WCAGNote                                                    │
│    ↓ HAS_REFERENCE                                           │
│  WCAGReference                                               │
│    ↔ RELATED_CRITERION / RELATED_GUIDELINE                   │
│  Cross-references between criteria                           │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

### Node Types

1. **WCAGPrinciple** (4 nodes)
   - Perceivable, Operable, Understandable, Robust
   - Top-level accessibility principles

2. **WCAGGuideline** (13 nodes)
   - Guidelines like "1.1 Text Alternatives", "2.1 Keyboard Accessible"
   - Organized under principles

3. **WCAGCriterion** (86+ nodes)
   - Individual success criteria (e.g., "1.1.1 Non-text Content")
   - Core testable requirements with levels A, AA, AAA

4. **ConformanceLevel** (3 nodes)
   - A: Minimum conformance
   - AA: Target for most organizations
   - AAA: Highest level

5. **WCAGSpecialCase**
   - Exceptions and special conditions per criterion

6. **WCAGNote**
   - Clarifying notes and additional context

7. **WCAGReference**
   - Links to "How to Meet" and "Understanding" documentation

### Relationship Types

- `PART_OF`: Hierarchical relationships (Criterion → Guideline → Principle)
- `HAS_LEVEL`: Links criteria to conformance levels
- `HAS_SPECIAL_CASE`: Links to exceptions
- `HAS_NOTE`: Links to clarifying notes
- `HAS_REFERENCE`: Links to external documentation
- `RELATED_CRITERION`: Cross-references between related criteria
- `RELATED_GUIDELINE`: Cross-references to related guidelines

---

## 📊 Neo4j Knowledge Graph vs Traditional Databases

### Traditional Relational Databases (SQL)

| Aspect | Traditional RDBMS | Limitations for This Use Case |
|--------|------------------|-------------------------------|
| **Data Model** | Tables with fixed schemas, foreign keys | Complex joins for multi-hop relationships |
| **Relationships** | Implicit via foreign keys | N+1 query problems for deep hierarchies |
| **Queries** | SQL with JOINs | Performance degrades with relationship depth |
| **Flexibility** | Schema changes require migrations | Difficult to add new relationship types |
| **Graph Traversal** | Recursive CTEs or multiple JOINs | Exponentially slower for 3+ levels |

**Example Query Complexity:**
```sql
-- Finding all criteria related to a specific principle (3 levels deep)
SELECT c.* FROM criteria c
JOIN guidelines g ON c.guideline_id = g.id
JOIN principles p ON g.principle_id = p.id
JOIN criterion_references cr ON c.id = cr.criterion_id
JOIN related_criteria rc ON rc.source_id = c.id
WHERE p.ref_id = '1';
-- Requires 5 JOINs, performance issues with deep traversals
```

### Neo4j Knowledge Graph

| Aspect | Neo4j Graph Database | Advantages for WCAG/GraphRAG |
|--------|---------------------|------------------------------|
| **Data Model** | Nodes and relationships as first-class citizens | Natural representation of WCAG hierarchy |
| **Relationships** | Explicit, typed, bidirectional | Direct navigation of complex relationships |
| **Queries** | Cypher query language | Pattern matching, intuitive graph traversal |
| **Flexibility** | Schema-optional | Easy to add new relationship types |
| **Graph Traversal** | Index-free adjacency | Constant-time relationship traversal |

**Example Query Simplicity:**
```cypher
// Finding all criteria related to a specific principle (3 levels deep)
MATCH (p:WCAGPrinciple {ref_id: '1'})<-[:PART_OF*1..2]-(c:WCAGCriterion)
OPTIONAL MATCH (c)-[:RELATED_CRITERION]->(related)
RETURN c, related;
// Single query, constant-time performance regardless of depth
```

### Key Differences

1. **Relationship Performance**
   - **RDBMS**: O(n log n) with indexes, degrades with JOINs
   - **Neo4j**: O(1) relationship traversal via index-free adjacency

2. **Schema Evolution**
   - **RDBMS**: Requires ALTER TABLE, migrations, downtime
   - **Neo4j**: Add properties/relationships without schema changes

3. **Query Intuition**
   - **RDBMS**: Think in tables and joins
   - **Neo4j**: Think in graph patterns (closer to human reasoning)

4. **WCAG Hierarchy Navigation**
   - **RDBMS**: Multiple self-joins for parent-child relationships
   - **Neo4j**: Single `MATCH` pattern with variable-length paths

5. **Cross-References**
   - **RDBMS**: Junction tables, complex queries
   - **Neo4j**: Direct relationship properties

---

## 🤖 GraphRAG vs Traditional LLM Training

### Traditional LLM Training Approaches

#### 1. **Fine-tuning**
```
┌─────────────┐      ┌──────────────┐      ┌──────────┐
│ WCAG Docs   │ ───▶ │ Training Set │ ───▶ │ Fine-tuned│
│ (Text)      │      │ (Examples)   │      │ LLM Model │
└─────────────┘      └──────────────┘      └──────────┘
```

**Limitations:**
- ❌ **Static Knowledge**: Model frozen at training time
- ❌ **Hallucination Risk**: May generate incorrect WCAG references
- ❌ **High Cost**: Requires retraining for updates (WCAG 2.3, 3.0)
- ❌ **No Traceability**: Cannot cite specific success criteria
- ❌ **Context Loss**: Hierarchical relationships flattened into text
- ❌ **Expensive**: GPU compute for training, frequent retraining needed

#### 2. **Embedding-based RAG (Vector Search)**
```
┌─────────────┐      ┌──────────────┐      ┌──────────┐
│ WCAG Docs   │ ───▶ │ Text Chunks  │ ───▶ │ Vector DB│
│             │      │ + Embeddings │      │ (Pinecone)│
└─────────────┘      └──────────────┘      └──────────┘
                                                  │
                     ┌──────────────┐            │
User Query ────────▶ │ Similarity   │ ◀──────────┘
                     │ Search       │
                     └──────────────┘
                            │
                     ┌──────▼────────┐
                     │ LLM Response  │
                     │ with Context  │
                     └───────────────┘
```

**Limitations:**
- ⚠️ **Semantic Only**: Finds similar text, not structural relationships
- ⚠️ **No Hierarchy**: Cannot traverse Principle → Guideline → Criterion
- ⚠️ **Chunking Issues**: May split related content across chunks
- ⚠️ **No Graph Reasoning**: Cannot answer "show all Level AA criteria under Principle 1"

### GraphRAG with Neo4j Knowledge Graph

```
┌─────────────┐      ┌──────────────┐      ┌──────────────┐
│ WCAG JSON   │ ───▶ │ ETL Pipeline │ ───▶ │ Neo4j Graph  │
│ (Structured)│      │ (This Project)│     │ (Structured) │
└─────────────┘      └──────────────┘      └──────────────┘
                                                    │
                     ┌──────────────────────────────┤
                     │                              │
              ┌──────▼─────────┐           ┌───────▼────────┐
User Query ──▶│ Cypher Query   │           │ Semantic Search│
              │ (Structural)   │           │ (If needed)    │
              └────────┬───────┘           └───────┬────────┘
                       │                           │
                       └────────┬──────────────────┘
                                │
                         ┌──────▼────────┐
                         │ Context with  │
                         │ Graph Paths + │
                         │ Relationships │
                         └──────┬────────┘
                                │
                         ┌──────▼────────┐
                         │ LLM Response  │
                         │ (Grounded)    │
                         └───────────────┘
```

### GraphRAG Advantages

#### 1. **Structured Knowledge Retrieval**
```cypher
// Find all Level AA criteria that are related to keyboard accessibility
MATCH (g:WCAGGuideline {ref_id: '2.1'})<-[:PART_OF]-(c:WCAGCriterion)
      -[:HAS_LEVEL]->(cl:ConformanceLevel {name: 'AA'})
OPTIONAL MATCH (c)-[:RELATED_CRITERION]->(related)
RETURN c, related;
```
- ✅ Precise, deterministic results
- ✅ Full relationship context
- ✅ No hallucination risk

#### 2. **Multi-hop Reasoning**
```cypher
// Find all special cases for criteria under the Perceivable principle
MATCH (p:WCAGPrinciple {name: 'Perceivable'})
      <-[:PART_OF*1..2]-(c:WCAGCriterion)
      -[:HAS_SPECIAL_CASE]->(sc:WCAGSpecialCase)
RETURN p.name, c.ref_id, c.title, sc.type, sc.description;
```
- ✅ Traverses arbitrary relationship depths
- ✅ Maintains hierarchical context

#### 3. **Explainable Results**
- ✅ **Graph Paths**: Show exact relationship chains
- ✅ **Citations**: Link to specific WCAG criteria with URLs
- ✅ **Traceability**: Audit trail for compliance decisions

#### 4. **Dynamic Updates**
- ✅ Update graph without retraining LLM
- ✅ Add new WCAG versions incrementally
- ✅ Link to JIRA bugs (future: `Bug -[:VIOLATES]-> WCAGCriterion`)

#### 5. **Hybrid Search**
```python
# Combine graph structure + semantic search
# 1. Use graph query to find structural matches
# 2. Use embeddings for semantic similarity within results
# 3. Combine scores for optimal relevance
```

#### 6. **Rich Context for LLM**
Traditional RAG context:
```
"1.1.1 Non-text Content (Level A): All non-text content..."
```

GraphRAG context:
```
Criterion: 1.1.1 Non-text Content (Level A)
Principle: Perceivable
Guideline: 1.1 Text Alternatives
Special Cases:
  - Controls/Input: If non-text content is a control...
  - Time-Based Media: If non-text content is time-based media...
Notes:
  - CAPTCHAs should provide alternative forms...
Related Criteria:
  - 4.1.2 Name, Role, Value (for controls)
  - 1.2.1 Audio-only and Video-only (for media)
References:
  - How to Meet: https://www.w3.org/WAI/WCAG22/quickref/#non-text-content
  - Understanding: https://www.w3.org/WAI/WCAG22/Understanding/non-text-content
```

### Comparison Table: GraphRAG vs Traditional Approaches

| Feature | Fine-tuning | Vector RAG | **GraphRAG (This Project)** |
|---------|-------------|------------|---------------------------|
| **Knowledge Updates** | Retrain required | Reindex required | ✅ Update graph only |
| **Hallucination Risk** | High | Medium | ✅ Low (grounded in graph) |
| **Relationship Queries** | ❌ Not possible | ⚠️ Limited | ✅ Native support |
| **Hierarchy Traversal** | ❌ Flattened | ⚠️ Chunked | ✅ Preserved |
| **Explainability** | ❌ Black box | ⚠️ Similarity scores | ✅ Graph paths + citations |
| **Cost** | $$$$ (GPU training) | $$ (embeddings) | ✅ $ (query only) |
| **Query Precision** | Low (probabilistic) | Medium (semantic) | ✅ High (structured) |
| **WCAG Updates** | Retrain model | Reembed chunks | ✅ Add/update nodes |
| **Cross-references** | ❌ Lost | ❌ Lost | ✅ Explicit relationships |
| **Conformance Levels** | ❌ Embedded in text | ⚠️ Embedded | ✅ First-class nodes |

---

## 🚀 Getting Started

### Prerequisites

- **Python 3.8+**
- **Neo4j Database** (AuraDB, Desktop, or self-hosted)
- **WCAG 2.2 JSON data** (`wcag_22_guidelines.json`)

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/jazxii/Neo4J-GraphRAG-ETL-pipeline.git
cd Neo4J-GraphRAG-ETL-pipeline
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Configure environment variables**
```bash
cp .env.example .env
# Edit .env with your Neo4j credentials
```

4. **Set up your `.env` file**
```env
NEO4J_URI=neo4j+s://your-instance.databases.neo4j.io
NEO4J_USER=neo4j
NEO4J_PASSWORD=your-password-here
WCAG_JSON_FILE=wcag_22_guidelines.json
```

### Running the Pipeline

```bash
python 01_pipeline_wcag_foundation.py
```

**Expected Output:**
```
🧹 Cleaning existing WCAG nodes (if any)...
   ✅ Cleaned auxiliary WCAG nodes

📐 Creating constraints and indexes...
   ✅ Constraints and indexes ready

🏷️  Creating conformance level nodes...
   ✅ Conformance levels: A, AA, AAA

📂 Loading wcag_22_guidelines.json...
   ✅ Loaded 4 principles

   📌 Principle 1: Perceivable
      📋 Guideline 1.1: Text Alternatives
         ✅ 1.1.1 Non-text Content (Level A)
      ...

🔗 Creating cross-reference relationships...
   ✅ 1.3.3 ──RELATED_GUIDELINE──▶ 1.4
   ...

============================================================
📊 WCAG FOUNDATION — VERIFICATION
============================================================

   Neo4j Node Counts:
   WCAGCriterion             86
   WCAGGuideline             13
   WCAGPrinciple             4
   ConformanceLevel          3
   ...

✅ WCAG Foundation loaded successfully!
```

---

## 📖 Usage Examples

### Example 1: Find All Level AA Criteria

```cypher
MATCH (c:WCAGCriterion)-[:HAS_LEVEL]->(cl:ConformanceLevel {name: 'AA'})
RETURN c.ref_id, c.title, c.description
ORDER BY c.ref_id;
```

### Example 2: Navigate the Hierarchy

```cypher
MATCH path = (c:WCAGCriterion {ref_id: '1.4.3'})
             -[:PART_OF]->(g:WCAGGuideline)
             -[:PART_OF]->(p:WCAGPrinciple)
RETURN path;
```

### Example 3: Find Related Criteria

```cypher
MATCH (c:WCAGCriterion {ref_id: '1.1.1'})
      -[:RELATED_CRITERION|RELATED_GUIDELINE]-(related)
RETURN c.title, type(r) as relationship, related.ref_id, related.title;
```

### Example 4: Get Complete Context for LLM

```cypher
MATCH (c:WCAGCriterion {ref_id: '1.1.1'})
OPTIONAL MATCH (c)-[:PART_OF]->(g:WCAGGuideline)-[:PART_OF]->(p:WCAGPrinciple)
OPTIONAL MATCH (c)-[:HAS_LEVEL]->(cl:ConformanceLevel)
OPTIONAL MATCH (c)-[:HAS_SPECIAL_CASE]->(sc:WCAGSpecialCase)
OPTIONAL MATCH (c)-[:HAS_NOTE]->(n:WCAGNote)
OPTIONAL MATCH (c)-[:HAS_REFERENCE]->(r:WCAGReference)
RETURN c, g, p, cl, collect(sc) as special_cases, 
       collect(n) as notes, collect(r) as references;
```

---

## 🔮 Future Enhancements

### Phase 2: JIRA Integration
```cypher
// Link accessibility bugs to WCAG violations
CREATE (bug:JIRAIssue {key: 'ACC-123', summary: '...'})
CREATE (bug)-[:VIOLATES]->(c:WCAGCriterion {ref_id: '1.1.1'})
```

### Phase 3: GraphRAG Query Engine
```python
# Hybrid retrieval: Graph structure + Semantic search
def graphrag_query(user_question):
    # 1. Parse intent
    # 2. Execute Cypher query for structure
    # 3. Retrieve graph context
    # 4. Augment with embeddings if needed
    # 5. Generate LLM response with citations
    return response_with_graph_context
```

### Phase 4: LLM Training Dataset
```python
# Generate training examples from graph
# Each path = a training sample with rich context
MATCH path = (c:WCAGCriterion)-[*1..3]-(related)
RETURN path
// Convert to instruction-tuning format
```

---

## 📁 Project Structure

```
Neo4J-GraphRAG-ETL-pipeline/
├── 01_pipeline_wcag_foundation.py  # Main ETL script
├── wcag_22_guidelines.json         # WCAG 2.2 source data
├── requirements.txt                # Python dependencies
├── .env.example                    # Environment template
├── .env                            # Your credentials (gitignored)
├── .gitignore                      # Git ignore rules
└── README.md                       # This file
```

---

## 🛠️ Technical Details

### Constraints and Indexes

The pipeline creates:
- **Unique constraints** on `ref_id` for Principles, Guidelines, Criteria
- **Unique constraint** on `name` for ConformanceLevel
- **Indexes** on commonly queried properties (code, title, level)

### Data Integrity

- ✅ **Idempotent**: Safe to run multiple times (uses `MERGE`)
- ✅ **Referential Integrity**: All relationships validated
- ✅ **Verification**: Built-in checks for broken chains

### Performance

- ⚡ **Index-free adjacency**: O(1) relationship traversal
- ⚡ **Batch loading**: Efficient bulk operations
- ⚡ **Query optimization**: Indexed properties for fast lookups

---

## 📚 Resources

### WCAG 2.2 Specification
- [Official Specification](https://www.w3.org/TR/WCAG22/)
- [Quick Reference](https://www.w3.org/WAI/WCAG22/quickref/)
- [Understanding WCAG 2.2](https://www.w3.org/WAI/WCAG22/Understanding/)

### Neo4j Documentation
- [Neo4j Documentation](https://neo4j.com/docs/)
- [Cypher Query Language](https://neo4j.com/docs/cypher-manual/current/)
- [Graph Data Science](https://neo4j.com/docs/graph-data-science/current/)

### GraphRAG Research
- [Microsoft GraphRAG](https://microsoft.github.io/graphrag/)
- [Knowledge Graphs for RAG](https://arxiv.org/abs/2404.16130)
- [Neo4j + LLMs](https://neo4j.com/developer-blog/knowledge-graphs-llms-multi-hop-question-answering/)

---

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- Additional WCAG 2.3 / 3.0 support
- GraphRAG query engine implementation
- Embedding integration for hybrid search
- JIRA/Confluence connectors

---

## 📄 License

This project is open source. WCAG 2.2 specification © W3C.

---

## 👤 Author

**jazxii**
- GitHub: [@jazxii](https://github.com/jazxii)

---

## 🙏 Acknowledgments

- W3C for WCAG 2.2 specification
- Neo4j team for graph database technology
- GraphRAG research community