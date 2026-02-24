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

The pipeline creates a hierarchical, multi-layered knowledge graph.
When run against the **enriched JSON** (`wcag_22_guidelines_enriched.json`),
it additionally loads techniques, test rules, examples, and benefits.

```
┌────────────────────────────────────────────────────────────────────┐
│                    WCAG Knowledge Graph (Enriched)                  │
├────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  WCAGPrinciple (4)                                                  │
│    ↑ PART_OF                                                        │
│  WCAGGuideline (13)                                                 │
│    ↑ PART_OF                                                        │
│  WCAGCriterion (86+) ──→ HAS_LEVEL ──→ ConformanceLevel (3)        │
│    │                                                                 │
│    ├─ HAS_SPECIAL_CASE ──→ WCAGSpecialCase                          │
│    ├─ HAS_NOTE ──→ WCAGNote                                         │
│    ├─ HAS_REFERENCE ──→ WCAGReference                               │
│    ├─ HAS_TECHNIQUE ──→ WCAGTechnique  (sufficient)                 │
│    ├─ HAS_ADVISORY_TECHNIQUE ──→ WCAGTechnique  (advisory)          │
│    ├─ HAS_FAILURE ──→ WCAGTechnique  (failure patterns)             │
│    ├─ HAS_TEST_RULE ──→ WCAGTestRule  (ACT automated tests)         │
│    ├─ HAS_EXAMPLE ──→ WCAGExample                                   │
│    ├─ HAS_BENEFIT ──→ WCAGBenefit                                   │
│    └─ RELATED_CRITERION / RELATED_GUIDELINE ──→ cross-refs          │
│                                                                      │
│  Enriched properties on WCAGCriterion:                              │
│    intent, wcag_version, automatable, disability_impact,            │
│    input_types_affected, technology_applicability,                   │
│    in_brief_goal, in_brief_what_to_do, in_brief_why_important       │
│                                                                      │
└────────────────────────────────────────────────────────────────────┘
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
   - **Enriched properties**: `intent`, `wcag_version` (2.0/2.1/2.2), `automatable` (full/partial/manual), `disability_impact` (array), `input_types_affected` (array), `technology_applicability` (array), `in_brief_goal`, `in_brief_what_to_do`, `in_brief_why_important`

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

8. **WCAGTechnique** *(enriched)*
   - W3C techniques for meeting success criteria (e.g., G94, H37, ARIA6, F3)
   - Properties: `tech_id`, `title`, `url`, `technology` (html/css/aria/pdf/script), `category` (sufficient/advisory/failures)
   - Deduplicated globally — shared across multiple criteria

9. **WCAGTestRule** *(enriched)*
   - ACT (Accessibility Conformance Testing) rules for automated testing
   - Properties: `title`, `url`, `criterion_id`

10. **WCAGExample** *(enriched)*
    - Concrete examples illustrating how to satisfy each criterion
    - Properties: `title`, `description`, `criterion_id`, `index`

11. **WCAGBenefit** *(enriched)*
    - Describes who benefits from each criterion and how
    - Properties: `description`, `criterion_id`, `index`

### Relationship Types

- `PART_OF`: Hierarchical relationships (Criterion → Guideline → Principle)
- `HAS_LEVEL`: Links criteria to conformance levels
- `HAS_SPECIAL_CASE`: Links to exceptions
- `HAS_NOTE`: Links to clarifying notes
- `HAS_REFERENCE`: Links to external documentation
- `RELATED_CRITERION`: Cross-references between related criteria
- `RELATED_GUIDELINE`: Cross-references to related guidelines
- `HAS_TECHNIQUE` *(enriched)*: Sufficient techniques to pass a criterion
- `HAS_ADVISORY_TECHNIQUE` *(enriched)*: Advisory (recommended) techniques
- `HAS_FAILURE` *(enriched)*: Known failure patterns (F-series techniques)
- `HAS_TEST_RULE` *(enriched)*: ACT automated test rules
- `HAS_EXAMPLE` *(enriched)*: Illustrative examples
- `HAS_BENEFIT` *(enriched)*: Accessibility benefit descriptions

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

GraphRAG context (enriched):
```
Criterion: 1.1.1 Non-text Content (Level A)
Principle: Perceivable
Guideline: 1.1 Text Alternatives
WCAG Version: 2.0 | Automatable: partial
Disability Impact: blindness, low_vision, deafblindness, cognitive, learning
In Brief:
  Goal: Non-text information is available to more people
  What to do: Provide text alternatives for non-text content
  Why: Essential for screen readers and alternative presentations
Intent: If all non-text content is available as text, it can be
  rendered in ways adapted to any user's sensory abilities...
Sufficient Techniques: G94, ARIA6, ARIA10, H37, H36, H2...
Failure Patterns: F3, F13, F20, F30, F38, F39, F65, F67, F71, F72
ACT Test Rules: Image has non-empty accessible name
Special Cases:
  - Controls/Input: If non-text content is a control...
  - Time-Based Media: If non-text content is time-based media...
Notes:
  - CAPTCHAs should provide alternative forms...
Related Criteria: 1.2.1, 4.1.2
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

#### Option A: Base JSON (quick, no internet needed)
```bash
python 01_pipeline_wcag_foundation.py
```

#### Option B: Enriched JSON (recommended for rule engines)
```bash
# Step 1: Enrich the JSON by scraping W3C Understanding pages (~2 min)
python 00_enrich_wcag_json.py

# Step 2: Update .env to point to the enriched file
# WCAG_JSON_FILE=wcag_22_guidelines_enriched.json

# Step 3: Run the pipeline
python 01_pipeline_wcag_foundation.py
```

The enrichment script (`00_enrich_wcag_json.py`) scrapes each criterion's
[W3C Understanding page](https://www.w3.org/WAI/WCAG22/Understanding/) to extract:

| Enriched Field | Source | Rule Engine Value |
|---|---|---|
| `intent` | Understanding page | Explains *why* the criterion exists |
| `in_brief` | "In Brief" sidebar | Quick goal / what-to-do / why-it-matters |
| `techniques.sufficient` | Techniques section | Known ways to **pass** |
| `techniques.advisory` | Techniques section | Recommended enhancements |
| `techniques.failures` | Techniques section | Known ways to **fail** (F-series) |
| `test_rules` | Test Rules section | ACT rules for automation |
| `examples` | Examples section | Concrete illustrations |
| `benefits` | Benefits section | Disability groups helped |
| `wcag_version` | Derived | 2.0 / 2.1 / 2.2 |
| `automatable` | Classified | full / partial / manual |
| `disability_impact` | Mapped | blindness, cognitive, motor, etc. |
| `input_types_affected` | Classified | keyboard, pointer, visual, auditory |
| `related_scs` | Intent cross-refs | Dynamically discovered SC↔SC links |

**Expected Output (enriched):**
```
==============================================================
  WCAG 2.2 Foundation — ETL Pipeline
==============================================================
PHASE 0 ▸ Pre-flight checks
  Cleaned auxiliary WCAG nodes
  Constraints and indexes ready
PHASE 1 ▸ Extract — reading wcag_22_guidelines_enriched.json
  Validated 4 principles — structure OK
PHASE 2 ▸ Transform — normalizing records
  Detected ENRICHED JSON — extracting techniques, examples, test rules, benefits
  Transformed → 4 principles, 13 guidelines, 86 criteria, ...
  Enriched    → 300+ techniques, 50+ test rules, 200+ examples, 100+ benefits
PHASE 3 ▸ Load — writing to Neo4j
  Loaded 3 conformance levels
  Loaded 4 principles
  Loaded 13 guidelines
  Loaded 86 criteria
  Loaded 300+ technique nodes
  Loaded 50+ test rules
  Loaded 200+ examples
  Loaded 100+ benefits
PHASE 4 ▸ Validate — integrity checks
  PASS — All 86 criteria have complete hierarchy chains
  PASS — WCAG version distribution: 2.0 (50), 2.1 (27), 2.2 (9)
  PASS — Automation classification: full (12), partial (35), manual (39)
✅ All validation checks passed
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

### Example 5: Find Techniques to Pass a Criterion (Enriched)

```cypher
MATCH (c:WCAGCriterion {ref_id: '1.4.3'})-[:HAS_TECHNIQUE]->(t:WCAGTechnique)
RETURN t.tech_id, t.title, t.technology
ORDER BY t.technology;
// Returns: G18, G145, G148 (general), C21, C22 (css), etc.
```

### Example 6: Find All Known Failure Patterns (Rule Engine)

```cypher
MATCH (c:WCAGCriterion)-[:HAS_FAILURE]->(f:WCAGTechnique)
WHERE c.level IN ['A', 'AA']
RETURN c.ref_id, c.title, collect(f.tech_id) AS failure_ids
ORDER BY c.ref_id;
// Powers automated compliance scanning
```

### Example 7: Query by Automation Level

```cypher
// Find all fully automatable Level AA criteria with their test rules
MATCH (c:WCAGCriterion {automatable: 'full', level: 'AA'})
OPTIONAL MATCH (c)-[:HAS_TEST_RULE]->(tr:WCAGTestRule)
RETURN c.ref_id, c.title, collect(tr.title) AS act_rules;
```

### Example 8: Disability Impact Analysis

```cypher
// Which criteria affect users with cognitive disabilities?
MATCH (c:WCAGCriterion)
WHERE 'cognitive' IN c.disability_impact
RETURN c.ref_id, c.title, c.level, c.wcag_version
ORDER BY c.level, c.ref_id;
```

### Example 9: Full Rule Engine Context for a Criterion

```cypher
MATCH (c:WCAGCriterion {ref_id: '2.4.7'})
OPTIONAL MATCH (c)-[:HAS_TECHNIQUE]->(suf:WCAGTechnique)
OPTIONAL MATCH (c)-[:HAS_FAILURE]->(fail:WCAGTechnique)
OPTIONAL MATCH (c)-[:HAS_TEST_RULE]->(tr:WCAGTestRule)
OPTIONAL MATCH (c)-[:HAS_EXAMPLE]->(ex:WCAGExample)
OPTIONAL MATCH (c)-[:HAS_BENEFIT]->(b:WCAGBenefit)
RETURN c.ref_id, c.title, c.intent, c.automatable,
       c.in_brief_goal, c.in_brief_what_to_do,
       collect(DISTINCT suf.tech_id) AS sufficient_techniques,
       collect(DISTINCT fail.tech_id) AS failure_patterns,
       collect(DISTINCT tr.title) AS test_rules,
       collect(DISTINCT ex.title) AS examples,
       collect(DISTINCT b.description) AS benefits;
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
├── 00_enrich_wcag_json.py              # Step 0: Enrich JSON from W3C Understanding pages
├── 01_pipeline_wcag_foundation.py      # Step 1: Main ETL pipeline (base or enriched)
├── wcag_22_guidelines.json             # WCAG 2.2 base source data
├── wcag_22_guidelines_enriched.json    # WCAG 2.2 enriched data (generated by Step 0)
├── requirements.txt                    # Python dependencies
├── .env.example                        # Environment template
├── .env                                # Your credentials (gitignored)
├── .gitignore                          # Git ignore rules
└── README.md                           # This file
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