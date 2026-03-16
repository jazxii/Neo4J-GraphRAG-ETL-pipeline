# Neo4j Agentic GraphRAG for WCAG 2.2

> **An AI Agent that reasons over a WCAG 2.2 Knowledge Graph to answer accessibility compliance questions — powered by Neo4j, a 5-phase ETL pipeline, and a ReAct (Reasoning + Acting) agent with 9 specialized tools, LLM-powered query decomposition, and dynamic Cypher generation with guardrails.**

---

## 🎯 Project Overview

This project builds an **Agentic RAG (Retrieval-Augmented Generation)** system on top of a **Neo4j Knowledge Graph** containing the full WCAG 2.2 specification. Unlike traditional vector-only RAG, the agent can **reason over structured relationships**, traverse the WCAG hierarchy, find techniques, analyze disability impact, resolve terminology definitions, perform multi-hop cross-references, and assemble rich context — all before generating a response.

### What Makes This Different

| Approach | How It Works | Limitation |
|----------|-------------|------------|
| **Fine-tuning** | Bake WCAG into model weights | Frozen knowledge, hallucinations, no citations |
| **Vector RAG** | Embed chunks → similarity search | Loses hierarchy, can't traverse relationships |
| **GraphRAG** | Cypher queries over knowledge graph | Single-shot retrieval, no reasoning |
| **Agentic GraphRAG** ✅ | Agent plans, selects tools, traverses graph iteratively | — This project |

### The Pipeline

```
┌────────────────────────────────────────────────────────────────────────────┐
│                          END-TO-END PIPELINE                               │
│                                                                            │
│   Step 0               Step 1                  Step 2                     │
│  ┌───────────────┐    ┌─────────────────────┐  ┌────────────────────────┐ │
│  │  00_scrape     │    │  01_pipeline_wcag    │  │  02_agentic_rag_wcag  │ │
│  │  _wcag_to      │───▶│  _foundation.py     │─▶│  .py                  │ │
│  │  _csv.py       │    │                     │  │                        │ │
│  │ Scrape W3C    │    │ Extract → Transform │  │ ReAct Agent with      │ │
│  │ Understanding │    │ → Load → Validate   │  │ 9 tools + decomposer  │ │
│  │ pages         │    │ into Neo4j KG       │  │ over Neo4j KG         │ │
│  └───────────────┘    └─────────────────────┘  └────────────────────────┘ │
│         │                       │                         │                │
│         ▼                       ▼                         ▼                │
│  wcag_22_guidelines      ┌───────────┐         ┌──────────────────┐      │
│                          │  Neo4j    │◀────────│  User queries    │      │
│                          │  WCAG KG  │         │  answered with   │      │
│                          │ (13 node  │         │  graph-grounded  │      │
│                          │  types,   │         │  citations       │      │
│                          │  2,418    │         └──────────────────┘      │
│                          │  nodes)   │                                    │
│                          └───────────┘                                    │
└────────────────────────────────────────────────────────────────────────────┘
```

---

## 🤖 Agentic RAG System

### Architecture

The agent follows a **ReAct (Reasoning + Acting)** loop — it doesn't just retrieve, it *reasons* about what information it needs, calls the right tools, and iterates until it has enough context to answer.

```
                          User Query
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│                      WCAG AGENT (Orchestrator)                   │
│                                                                  │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │  0. DECOMPOSE — Break query into prioritized sub-steps   │   │
│   │  1. THINK   — Analyze each sub-step, plan tool calls     │   │
│   │  2. ACT     — Call one or more specialized tools         │   │
│   │  3. OBSERVE — Process results, resolve dependencies      │   │
│   │  4. REPEAT  — Loop until all sub-steps completed         │   │
│   │  5. RESPOND — Synthesize answer with graph citations     │   │
│   └─────────────────────────────────────────────────────────┘   │
│          │         │          │          │          │            │
│   ┌──────▼──┐ ┌───▼──────┐ ┌▼───────┐ ┌▼────────┐ ┌▼────────┐ │
│   │ Graph   │ │ Semantic │ │Technique│ │ Rule    │ │ Impact  │ │
│   │Traversal│ │ Search   │ │ Finder  │ │ Engine  │ │Analysis │ │
│   └─────────┘ └──────────┘ └────────┘ └─────────┘ └─────────┘ │
│   ┌─────────┐ ┌──────────┐ ┌──────────┐ ┌────────────────┐     │
│   │Key Term │ │ Cross-   │ │ Dynamic  │ │   Context      │     │
│   │ Lookup  │ │Reference │ │ Cypher   │ │  Assembler     │     │
│   └─────────┘ └──────────┘ │(LLM+guard│ └────────────────┘     │
│                             │  rails)  │                        │
│                             └──────────┘                        │
│                │               │                                │
│                └───────┬───────┘                                │
│                        ▼                                        │
│                 ┌─────────────┐                                 │
│                 │   Neo4j     │                                 │
│                 │  WCAG KG    │                                 │
│                 └─────────────┘                                 │
└──────────────────────────────────────────────────────────────────┘
                              │
                              ▼
              Graph-Grounded Response with Citations
```

### Two Operating Modes

The agent works **with or without an LLM**:

| Mode | Requirements | How It Decides Which Tool to Call |
|------|-------------|----------------------------------|
| **Rule-Based** | Neo4j only | QueryDecomposer (heuristic) + StepExecutor — pattern matching creates a prioritized plan |
| **LLM-Powered** | Neo4j + OpenAI / Ollama / Anthropic | QueryDecomposer (LLM) creates a plan → LLM uses function calling to execute it, with dynamic Cypher for ad-hoc queries |

Rule-based mode is ideal for testing, CI pipelines, and environments without LLM access. The LLM mode adds natural language understanding, dynamic Cypher generation, and more nuanced tool orchestration.

### 9 Specialized Tools

Each tool is backed by targeted Cypher queries against the WCAG Knowledge Graph:

| # | Tool | What It Does | Trigger Examples |
|---|------|-------------|-----------------|
| 1 | **`graph_traversal`** | Navigate the WCAG hierarchy (Principle → Guideline → Criterion) | "Show me guideline 2.1", "Get criterion 1.4.3" |
| 2 | **`semantic_search`** | Keyword search across criteria, examples, benefits, and key terms | "color contrast requirements", "keyboard navigation" |
| 3 | **`technique_finder`** | Find sufficient, advisory, and failure techniques per criterion | "How do I comply with 1.1.1?", "What failures exist for 2.4.7?" |
| 4 | **`rule_engine`** | Compliance checking by element type, disability, level, or WCAG version | "What rules apply to forms?", "What's new in WCAG 2.2?" |
| 5 | **`impact_analysis`** | Disability and input modality impact analysis | "Which criteria affect blind users?", "Keyboard-only requirements" |
| 6 | **`key_term_lookup`** | Look up WCAG key term definitions from 725 formal terms | "What does programmatically determined mean?", "Define text alternative" |
| 7 | **`cross_reference`** | Multi-hop graph walks: related chains, shared techniques, disability overlap, ripple effects | "If I fix 2.1.1, what else improves?", "Overlap between blindness and motor?" |
| 8 | **`dynamic_cypher`** | LLM-generated Cypher with guardrails for ad-hoc analytical queries | "Which criteria have more than 5 techniques?", "Count techniques per technology" |
| 9 | **`context_assembler`** | Builds LLM-ready structured context from full graph subgraphs including key terms and related resources | Auto-called as final step to assemble response context |

### Agent Usage

```bash
# Interactive mode (works without LLM — rule-based routing)
python 02_agentic_rag_wcag.py --interactive

# Single query
python 02_agentic_rag_wcag.py --query "What WCAG criteria apply to images?"

# With verbose reasoning trace
python 02_agentic_rag_wcag.py -v --query "How do I make forms accessible?"

# With LLM-powered routing (set OPENAI_API_KEY in .env)
python 02_agentic_rag_wcag.py --interactive
```

### Example Queries

```
🔍 What WCAG criteria apply to images?
🔍 How do I comply with 1.4.3 contrast?
🔍 What criteria affect blind users?
🔍 Give me a Level AA conformance checklist
🔍 What's new in WCAG 2.2?
🔍 Which criteria can be automatically tested?
🔍 What techniques fix keyboard accessibility?
🔍 Show me the full WCAG hierarchy
🔍 What are the failure patterns for 1.1.1?
🔍 Which criteria impact cognitive disabilities?
🔍 What does programmatically determined mean?
🔍 If I fix keyboard issues for 2.1.1, what else benefits?
🔍 Which criteria overlap between color blindness and low vision?
🔍 What does technique H37 cover?
🔍 What shared techniques apply to both 1.1.1 and 4.1.2?
🔍 Which criteria have more than 5 techniques?    ← dynamic_cypher
🔍 Show me criteria WITHOUT test rules             ← dynamic_cypher
🔍 Count techniques per technology type             ← dynamic_cypher
🔍 Audit a login form with images, inputs, buttons ← multi-step decomposition
```

### Example Agent Traces

**Terminology lookup (2 steps, 8s):**
```
──────────────────────────────────────────────────
QUERY: What does programmatically determined mean?
──────────────────────────────────────────────────
  Step 1 [🔧 ACT]
    Tool: key_term_lookup({"term": "programmatically determined"})
    Result: ✅ Found definition + 26 criteria that use this term

  Step 2 [💬 RESPOND]
    📖 programmatically determined: determined by software from
       author-supplied data in a way that different user agents,
       including assistive technologies, can extract and present
       this information to users in various modalities.
    Used in: 1.1.1, 1.2.2, 1.3.1, 1.3.2, 1.3.5, ... (+21 more)

──────────────────────────────────────────────────
COMPLETED in 8.04s (2 steps, tools: [key_term_lookup])
──────────────────────────────────────────────────
```

**Ripple effect analysis (3 steps, 25s):**
```
──────────────────────────────────────────────────
QUERY: If I fix keyboard issues for 2.1.1, what else benefits?
──────────────────────────────────────────────────
  Step 1 [🔧 ACT]
    Tool: cross_reference({"analysis_type": "fix_ripple_effect", "criterion_id": "2.1.1"})
    Result: ✅ 14 affected criteria (9 directly related + 5 via shared techniques)

  Step 2 [🔧 ACT]
    Tool: context_assembler({"criterion_ids": ["1.3.1", "2.1.3", "2.4.7", "2.4.13", ...]})
    Result: ✅ Assembled full context for 14 criteria

  Step 3 [💬 RESPOND]
    Directly related: 2.4.7 Focus Visible, 3.2.1 On Focus, 4.1.2 Name/Role/Value...
    Shared techniques: 2.4.3 Focus Order, 2.4.4 Link Purpose...
    Total affected: 14 criteria across Levels A, AA, AAA

──────────────────────────────────────────────────
COMPLETED in 25.27s (3 steps, tools: [cross_reference, context_assembler])
──────────────────────────────────────────────────
```

**Element-based rule engine (4 steps, <1s):**
```
──────────────────────────────────────────────────
QUERY: How do I make images accessible?
──────────────────────────────────────────────────
  Step 1 [🧠 THINK]
    Intent: element_rules → tool_plan: [rule_engine(image)]

  Step 2 [🔧 ACT]
    Tool: rule_engine({"check_type": "element_rules", "element_type": "image"})
    Result: ✅ Found 3 criteria: 1.1.1, 1.4.5, 1.4.9

  Step 3 [🔧 ACT]
    Tool: context_assembler({"criterion_ids": ["1.1.1", "1.4.5", "1.4.9"]})
    Result: ✅ Assembled hierarchy + techniques + examples + benefits + key terms

  Step 4 [💬 RESPOND]
    ═══════════════════════════════════════
    WCAG 1.1.1: Non-text Content
    ═══════════════════════════════════════
    Techniques:
      ✅ [H37]  Using alt attributes on img elements
      ❌ [F65]  Failure: omitting alt attribute on img
    Key Terms:
      📖 text alternative: Text that is programmatically associated...
      📖 non-text content: Any content that is not a sequence of characters...
    Related Resources:
      🔗 WAI Web Accessibility Tutorials: Images — ...

──────────────────────────────────────────────────
COMPLETED in 0.34s (4 steps, tools: [rule_engine, context_assembler])
──────────────────────────────────────────────────
```

### Query Decomposition & Planning

Complex queries are automatically broken into **prioritized sub-steps** before execution:

```
User: "Audit a login form with images, inputs, and buttons"
                          │
                          ▼
              ┌─────────────────────────┐
              │   QueryDecomposer       │
              │  (LLM or heuristic)     │
              └────────┬────────────────┘
                       │
        ┌──────────────┼──────────────────┐
        ▼              ▼                  ▼
   Step 1 [P1]    Step 2 [P1]       Step 3 [P1]
   rule_engine     rule_engine       rule_engine
   (image rules)   (form rules)     (button rules)
        │              │                  │
        └──────────────┼──────────────────┘
                       ▼
               Step 4 [P2] (depends on 1,2,3)
               context_assembler
               (union of all criteria found)
                       │
                       ▼
               Synthesized Response
```

**How it works:**
1. **QueryDecomposer** breaks the query into steps, each with: tool name, parameters, dependencies, and priority
2. **StepExecutor** runs steps in dependency/priority order, passing intermediate results between steps
3. Results from earlier steps (e.g., criterion IDs) automatically flow into dependent steps
4. The LLM can deviate from the plan if intermediate results suggest a better path

### Dynamic Cypher Tool (Guardrails)

Tool 8 (`dynamic_cypher`) lets the LLM generate ad-hoc Cypher for questions the pre-built templates can't handle:

```
User: "Which criteria have more than 5 techniques?"
                          │
                          ▼
              ┌───────────────────────┐
              │  DynamicCypherTool    │
              │                       │
              │  1. LLM generates     │──→ MATCH (c:WCAGCriterion)-[:HAS_TECHNIQUE]->(t)
              │     Cypher            │    WITH c, count(t) AS cnt
              │                       │    WHERE cnt > 5
              │  2. Guardrail check   │    RETURN c.ref_id, c.title, cnt
              │     ├ READ-ONLY? ✅   │    ORDER BY cnt DESC
              │     ├ Schema OK? ✅   │    LIMIT 50
              │     ├ No APOC? ✅     │
              │     └ LIMIT added? ✅ │
              │                       │
              │  3. Execute safely    │
              └───────────────────────┘
```

**5 Guardrails enforced on every dynamic query:**

| Guard | What It Blocks | Example |
|-------|---------------|---------|
| **Mutation Keywords** | CREATE, DELETE, SET, MERGE, REMOVE, DROP, DETACH | `CREATE (n:Bad) RETURN n` → ❌ |
| **Node Label Allowlist** | Only 13 known WCAG node types | `MATCH (n:FakeNode)` → ❌ |
| **Relationship Allowlist** | Only 15 known relationship types | `[:UNKNOWN_REL]` → ❌ |
| **Procedure Block** | No APOC, GDS, db.*, dbms.* calls | `CALL apoc.meta.data()` → ❌ |
| **Result Limit** | LIMIT 50 auto-injected if missing | Prevents unbounded scans |

### Why Agentic RAG > Plain RAG

| Capability | Plain Vector RAG | Plain GraphRAG | **Agentic GraphRAG** |
|-----------|-----------------|----------------|---------------------|
| Multi-step reasoning | ❌ Single retrieval | ❌ Single query | ✅ Iterative tool loop |
| Query decomposition | ❌ | ❌ | ✅ LLM + heuristic planner |
| Dynamic Cypher | ❌ | ⚠️ Manual | ✅ LLM-generated with guardrails |
| Relationship traversal | ❌ Lost in chunks | ✅ Cypher queries | ✅ + Agent decides depth |
| Dynamic tool selection | ❌ Fixed pipeline | ❌ Fixed query | ✅ Agent plans per query |
| Technique lookup | ❌ Embedded in text | ⚠️ Manual query | ✅ Dedicated tool |
| Disability impact | ❌ Not structured | ⚠️ If you know Cypher | ✅ Natural language → analysis |
| Terminology resolution | ❌ Not indexed | ⚠️ Manual query | ✅ 725 key terms searchable |
| Cross-reference analysis | ❌ Lost | ⚠️ Manual multi-query | ✅ Multi-hop graph walks |
| Ripple effect mapping | ❌ Impossible | ⚠️ Complex Cypher | ✅ One-step tool call |
| Ad-hoc analytics | ❌ | ⚠️ Write Cypher manually | ✅ Natural language → safe Cypher |
| Reasoning transparency | ❌ Black box | ❌ No trace | ✅ Full `AgentTrace` log |
| Works without LLM | N/A (requires LLM) | ✅ (Cypher only) | ✅ Rule-based fallback |
| Context assembly | Token-limited chunks | Manual RETURN clause | ✅ Auto-assembled with key terms + resources |

---

## 🏗️ Knowledge Graph Architecture

### Graph Schema

The ETL pipeline creates a 13-node-type knowledge graph (2,418 nodes) with 15 relationship types:

```
┌──────────────────────────────────────────────────────────────────────┐
│                    WCAG Knowledge Graph (Enriched)                    │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  WCAGPrinciple (4)                                                   │
│    ↑ PART_OF                                                         │
│  WCAGGuideline (13)                                                  │
│    ↑ PART_OF                                                         │
│  WCAGCriterion (87) ──→ HAS_LEVEL ──→ ConformanceLevel (3)          │
│    │                                                                 │
│    ├─ HAS_SPECIAL_CASE ──→ WCAGSpecialCase (91)                      │
│    ├─ HAS_NOTE ──→ WCAGNote (41)                                     │
│    ├─ HAS_REFERENCE ──→ WCAGReference (184)                          │
│    ├─ HAS_TECHNIQUE ──→ WCAGTechnique (412) (sufficient)             │
│    ├─ HAS_ADVISORY_TECHNIQUE ──→ WCAGTechnique  (advisory)           │
│    ├─ HAS_FAILURE ──→ WCAGTechnique  (failure patterns)              │
│    ├─ HAS_TEST_RULE ──→ WCAGTestRule (109) (ACT automated tests)     │
│    ├─ HAS_EXAMPLE ──→ WCAGExample (284)                              │
│    ├─ HAS_BENEFIT ──→ WCAGBenefit (204)                              │
│    ├─ HAS_KEY_TERM ──→ WCAGKeyTerm (725)  ← NEW                     │
│    ├─ HAS_RELATED_RESOURCE ──→ WCAGRelatedResource (261) ← NEW      │
│    └─ RELATED_CRITERION / RELATED_GUIDELINE ──→ cross-refs           │
│                                                                      │
│  Enriched properties on WCAGCriterion:                               │
│    intent, wcag_version, automatable, disability_impact,             │
│    input_types_affected, technology_applicability,                    │
│    in_brief_goal, in_brief_what_to_do, in_brief_why_important        │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

### Node Types

| # | Node Label | Count | Description |
|---|-----------|-------|-------------|
| 1 | **WCAGPrinciple** | 4 | Perceivable, Operable, Understandable, Robust |
| 2 | **WCAGGuideline** | 13 | e.g., "1.1 Text Alternatives", "2.1 Keyboard Accessible" |
| 3 | **WCAGCriterion** | 87 | Testable success criteria with level A/AA/AAA. Enriched with `intent`, `wcag_version`, `automatable`, `disability_impact`, `input_types_affected`, `technology_applicability`, `in_brief_goal`, `in_brief_what_to_do`, `in_brief_why_important` |
| 4 | **ConformanceLevel** | 3 | A (minimum), AA (target), AAA (highest) |
| 5 | **WCAGSpecialCase** | 91 | Exceptions and special conditions per criterion |
| 6 | **WCAGNote** | 41 | Clarifying notes and additional context |
| 7 | **WCAGReference** | 184 | Links to "How to Meet" and "Understanding" docs |
| 8 | **WCAGTechnique** *(enriched)* | 412 | W3C techniques: `tech_id`, `title`, `url`, `technology` (html/css/aria/pdf/script), `category` (sufficient/advisory/failures). Globally deduplicated. |
| 9 | **WCAGTestRule** *(enriched)* | 109 | ACT rules for automated accessibility testing |
| 10 | **WCAGExample** *(enriched)* | 284 | Concrete examples illustrating compliance |
| 11 | **WCAGBenefit** *(enriched)* | 204 | Who benefits and how (per disability category) |
| 12 | **WCAGKeyTerm** *(enriched)* | 725 | Formal WCAG term definitions (e.g., "programmatically determined", "text alternative"). Queried by `key_term_lookup` tool. |
| 13 | **WCAGRelatedResource** *(enriched)* | 261 | External resources and related documentation. Surfaced in context assembler output. |

### Relationship Types

| Relationship | From → To | Description |
|-------------|----------|-------------|
| `PART_OF` | Criterion → Guideline → Principle | Hierarchy |
| `HAS_LEVEL` | Criterion → ConformanceLevel | Level A/AA/AAA |
| `HAS_SPECIAL_CASE` | Criterion → SpecialCase | Exceptions |
| `HAS_NOTE` | Criterion → Note | Clarifications |
| `HAS_REFERENCE` | Criterion/Guideline → Reference | Documentation links |
| `RELATED_CRITERION` | Criterion → Criterion | Cross-references |
| `RELATED_GUIDELINE` | Criterion → Guideline | Cross-references |
| `HAS_TECHNIQUE` *(enriched)* | Criterion → Technique | Sufficient techniques |
| `HAS_ADVISORY_TECHNIQUE` *(enriched)* | Criterion → Technique | Advisory techniques |
| `HAS_FAILURE` *(enriched)* | Criterion → Technique | Known failure patterns |
| `HAS_TEST_RULE` *(enriched)* | Criterion → TestRule | ACT automated tests |
| `HAS_EXAMPLE` *(enriched)* | Criterion → Example | Illustrative examples |
| `HAS_BENEFIT` *(enriched)* | Criterion → Benefit | Benefit descriptions |
| `HAS_KEY_TERM` *(enriched)* | Criterion → KeyTerm | Formal WCAG definitions |
| `HAS_RELATED_RESOURCE` *(enriched)* | Criterion → RelatedResource | External resources |

---

## ⚙️ ETL Pipeline

### Pipeline Phases

The ETL pipeline (`01_pipeline_wcag_foundation.py`) runs in 5 clearly separated phases:

```
┌────────┐   ┌─────────┐   ┌───────────┐   ┌────────┐   ┌──────────┐
│Phase 0 │──▶│ Phase 1 │──▶│  Phase 2  │──▶│Phase 3 │──▶│ Phase 4  │
│Preflight│  │ Extract │   │ Transform │   │ Load   │   │ Validate │
│        │   │         │   │           │   │        │   │          │
│Clean DB│   │Read JSON│   │Normalize  │   │Batch   │   │Integrity │
│Schema  │   │Validate │   │Flatten    │   │UNWIND  │   │Checks    │
│Indexes │   │Structure│   │Deduplicate│   │to Neo4j│   │Counts    │
└────────┘   └─────────┘   └───────────┘   └────────┘   └──────────┘
```

| Phase | Name | What It Does |
|-------|------|-------------|
| **0** | Pre-flight | Full DB wipe (clean reload), creates constraints and indexes |
| **1** | Extract | Reads WCAG JSON, validates structure (required keys, valid levels) |
| **2** | Transform | Flattens hierarchy into batch-ready records, deduplicates references, detects enriched fields |
| **3** | Load | Batch `UNWIND` writes to Neo4j (~18 transactions for all node/edge types) |
| **4** | Validate | Checks hierarchy chains, code/ref_id consistency, conformance level linkage, enriched data |

The pipeline **auto-detects** whether the source JSON is base or enriched and adjusts accordingly — enriched data yields 6 additional node types and 8 additional relationship types.

### Data Enrichment

The enrichment script (`00_scrape_wcag_to_csv.py`) scrapes each criterion's [W3C Understanding page](https://www.w3.org/WAI/WCAG22/Understanding/) to add:

| Enriched Field | Source | Value for Agentic RAG |
|---|---|---|
| `intent` | Understanding page | Explains *why* the criterion exists |
| `in_brief` | "In Brief" sidebar | Quick goal / what-to-do / why-it-matters |
| `techniques.sufficient` | Techniques section | Known ways to **pass** — agent's TechniqueFinderTool |
| `techniques.advisory` | Techniques section | Recommended enhancements |
| `techniques.failures` | Techniques section | Known ways to **fail** (F-series) — agent's RuleEngineTool |
| `test_rules` | Test Rules section | ACT rules — powers automation classification |
| `examples` | Examples section | Concrete illustrations — context assembler |
| `benefits` | Benefits section | Disability groups helped — impact analysis tool |
| `key_terms` | Key Terms section | Formal WCAG definitions (725 terms) — powers `key_term_lookup` tool |
| `related_resources` | Resources section | External references (261 resources) — surfaced in context assembler |
| `wcag_version` | Derived | 2.0 / 2.1 / 2.2 — version diff queries |
| `automatable` | Classified | full / partial / manual — automation tool |
| `disability_impact` | Mapped | blindness, cognitive, motor, etc. — impact analysis |
| `input_types_affected` | Classified | keyboard, pointer, visual, auditory |
| `related_scs` | Intent cross-refs | Dynamically discovered SC↔SC links |

### Pipeline Output

```
==============================================================
  WCAG 2.2 Foundation — ETL Pipeline
==============================================================
PHASE 0 ▸ Pre-flight checks
  Wiped ALL WCAG nodes for clean reload
  Constraints and indexes ready
PHASE 1 ▸ Extract — reading wcag_22_guidelines_enriched.json
  Validated 4 principles — structure OK
PHASE 2 ▸ Transform — normalizing records
  Detected ENRICHED JSON — extracting techniques, examples, test rules, benefits
  Transformed → 4 principles, 13 guidelines, 87 criteria, ...
  Enriched    → 412 techniques, 109 test rules, 284 examples, 204 benefits
  Key Terms   → 725 terms, Related Resources → 261 resources
PHASE 3 ▸ Load — writing to Neo4j
  Loaded 3 conformance levels
  Loaded 4 principles
  Loaded 13 guidelines
  Loaded 87 criteria
  Loaded 412 technique nodes
  Loaded 109 test rules
  Loaded 284 examples
  Loaded 204 benefits
  Loaded 725 key terms
  Loaded 261 related resources
PHASE 4 ▸ Validate — integrity checks
  PASS — All 87 criteria have complete hierarchy chains
  PASS — 2,418 total nodes across 13 node types
  PASS — WCAG version distribution: 2.0 (50), 2.1 (28), 2.2 (9)
  PASS — Automation classification: full (12), partial (35), manual (40)
✅ All validation checks passed (56.41s)
```

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

**Example — 3-level hierarchy query in SQL:**
```sql
SELECT c.* FROM criteria c
JOIN guidelines g ON c.guideline_id = g.id
JOIN principles p ON g.principle_id = p.id
JOIN criterion_references cr ON c.id = cr.criterion_id
JOIN related_criteria rc ON rc.source_id = c.id
WHERE p.ref_id = '1';
-- 5 JOINs, performance issues with deep traversals
```

### Neo4j Knowledge Graph

| Aspect | Neo4j Graph Database | Advantages for WCAG / Agentic RAG |
|--------|---------------------|-------------------------------------|
| **Data Model** | Nodes and relationships as first-class citizens | Natural representation of WCAG hierarchy |
| **Relationships** | Explicit, typed, bidirectional | Agent tools traverse directly |
| **Queries** | Cypher query language | Pattern matching, intuitive graph traversal |
| **Flexibility** | Schema-optional | Easy to add new relationship types |
| **Graph Traversal** | Index-free adjacency | Constant-time relationship traversal |

**Same query in Cypher:**
```cypher
MATCH (p:WCAGPrinciple {ref_id: '1'})<-[:PART_OF*1..2]-(c:WCAGCriterion)
OPTIONAL MATCH (c)-[:RELATED_CRITERION]->(related)
RETURN c, related;
// Single query, constant-time regardless of depth
```

### Key Differences

1. **Relationship Performance** — RDBMS: O(n log n) with JOINs → Neo4j: O(1) via index-free adjacency
2. **Schema Evolution** — RDBMS: ALTER TABLE + migrations → Neo4j: add properties/relationships freely
3. **Query Intuition** — RDBMS: think in tables → Neo4j: think in graph patterns
4. **WCAG Hierarchy** — RDBMS: recursive self-joins → Neo4j: `MATCH ... -[:PART_OF*1..2]->`
5. **Cross-References** — RDBMS: junction tables → Neo4j: direct relationship properties
6. **Agent Integration** — RDBMS: complex ORM mapping → Neo4j: Cypher queries map directly to tool parameters

---

## 🆚 Agentic GraphRAG vs Other LLM Approaches

### Fine-tuning
```
WCAG Docs → Training Examples → Fine-tuned LLM (frozen)
```
- ❌ Static knowledge — retrain for WCAG 2.3/3.0
- ❌ Hallucination risk — may cite non-existent criteria
- ❌ No traceability — can't show which criterion applies
- ❌ Expensive — GPU compute for training

### Embedding-based RAG (Vector Search)
```
WCAG Docs → Chunks + Embeddings → Vector DB → Similarity Search → LLM
```
- ⚠️ Semantic only — finds similar text, not structural relationships
- ⚠️ Hierarchy lost — can't traverse Principle → Guideline → Criterion
- ⚠️ Chunking issues — may split related content
- ⚠️ No reasoning — single-shot retrieval

### GraphRAG (Static)
```
WCAG JSON → ETL → Neo4j → Cypher Query → LLM Context → Response
```
- ✅ Structured retrieval via graph
- ⚠️ Requires pre-written Cypher per query type
- ⚠️ No dynamic tool selection

### Agentic GraphRAG (This Project) ✅
```
WCAG JSON → ETL → Neo4j → Agent [Think → Tools → Observe → Repeat] → Grounded Response
```
- ✅ Agent reasons about *what* to query
- ✅ Multi-step: traversal + techniques + impact in one query
- ✅ Graph-grounded — every claim citable to a criterion ID
- ✅ Dynamic updates — change graph, not model
- ✅ Works with or without LLM
- ✅ Full reasoning trace for audit

### Comparison Table

| Feature | Fine-tuning | Vector RAG | Static GraphRAG | **Agentic GraphRAG** |
|---------|-------------|------------|-----------------|---------------------|
| **Knowledge Updates** | Retrain | Reindex | Update graph | ✅ Update graph |
| **Hallucination Risk** | High | Medium | Low | ✅ Very Low |
| **Relationship Queries** | ❌ | ⚠️ Limited | ✅ | ✅ Agent-driven |
| **Hierarchy Traversal** | ❌ | ⚠️ Chunked | ✅ | ✅ Multi-hop |
| **Multi-step Reasoning** | ❌ | ❌ | ❌ | ✅ ReAct loop |
| **Tool Selection** | N/A | N/A | Manual | ✅ 8 dynamic tools |
| **Explainability** | ❌ Black box | ⚠️ Scores | ✅ Cypher | ✅ Full trace |
| **Cost** | $$$$ GPU | $$ Embeddings | $ Queries | ✅ $ Queries |
| **WCAG Updates** | Retrain | Re-embed | Add nodes | ✅ Add nodes |
| **Cross-references** | ❌ Lost | ❌ Lost | ✅ Explicit | ✅ Agent traverses multi-hop |
| **Disability Analysis** | ❌ | ❌ | Manual query | ✅ Dedicated tool + overlap |
| **Terminology** | ❌ | ⚠️ Might find | Manual query | ✅ 725 key terms indexed |
| **Ripple Effects** | ❌ | ❌ | Complex Cypher | ✅ One-step cross_reference |

### Context Quality: Agentic GraphRAG vs Vector RAG

**Vector RAG context:**
```
"1.1.1 Non-text Content (Level A): All non-text content..."
```

**Agentic GraphRAG context (auto-assembled by agent):**
```
═══════════════════════════════════════
WCAG 1.1.1: Non-text Content
═══════════════════════════════════════
Principle:  Perceivable
Guideline:  1.1 Text Alternatives
Level:      A
WCAG Version: 2.0 | Automatable: partial
Disability Impact: blindness, low_vision, deafblindness, cognitive, learning

In Brief:
  Goal: Non-text information is available to more people
  What to do: Provide text alternatives for non-text content
  Why: Essential for screen readers and alternative presentations

Intent:
  If all non-text content is available as text, it can be rendered
  in ways adapted to any user's sensory abilities...

Techniques:
  ✅ [G94]  Short text alternative providing descriptive info
  ✅ [H37]  Using alt attributes on img elements
  ✅ [ARIA6] Using aria-label for objects
  💡 [C18]  Using CSS margin and padding rules for layout (advisory)
  ❌ [F3]   Failure: using CSS to include images of important info
  ❌ [F65]  Failure: omitting alt attribute on img

Automated Test Rules:
  🧪 Image has non-empty accessible name

Special Cases:
  • [exception] Controls/Input: If non-text content is a control...
  • [exception] Time-Based Media: If non-text content is time-based media...

Key Terms:
  📖 text alternative: Text that is programmatically associated with non-text...
  📖 non-text content: Any content that is not a sequence of characters...
  📖 assistive technology: Hardware and/or software that acts as a user agent...

Related Resources:
  🔗 WAI Web Accessibility Tutorials: Images — https://www.w3.org/WAI/tutorials/images/
  🔗 HTML5: Requirements for alt text — https://www.w3.org/TR/html5/...

Related Criteria:
  → 1.2.1 Audio-only and Video-only (Level A)
  → 4.1.2 Name, Role, Value (Level A)
```

---

## 🚀 Getting Started

### Prerequisites

- **Python 3.11+**
- **Neo4j 5.x** (AuraDB, Desktop, or Docker)
- *(Optional)* OpenAI API key or local Ollama for LLM-powered agent mode

### Installation

```bash
# 1. Clone
git clone https://github.com/jazxii/Neo4J-GraphRAG-ETL-pipeline.git
cd Neo4J-GraphRAG-ETL-pipeline

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure environment
cp .env.example .env
# Edit .env with your Neo4j credentials
```

### Running the Full Pipeline

```bash
# Step 0 (optional but recommended): Scrape & enrich WCAG data from W3C (~2 min)
python 00_scrape_wcag_to_csv.py

# Step 1: Load into Neo4j Knowledge Graph (2,418 nodes)
# (update WCAG_JSON_FILE in .env to use enriched version)
python 01_pipeline_wcag_foundation.py

# Step 2: Start the Agentic RAG (8 tools)
python 02_agentic_rag_wcag.py --interactive
```

### Environment Configuration

```env
# ── Neo4j ──
NEO4J_URI=neo4j+s://your-instance.databases.neo4j.io
NEO4J_USER=neo4j
NEO4J_PASSWORD=your-password-here

# ── WCAG Source ──
WCAG_JSON_FILE=wcag_22_guidelines_enriched.json

# ── Agentic RAG (optional — rule-based mode works without these) ──
LLM_PROVIDER=openai          # openai | ollama | anthropic
LLM_MODEL=gpt-4o
OPENAI_API_KEY=sk-your-key

# For Ollama (local, free):
# LLM_BASE_URL=http://localhost:11434/v1
# LLM_MODEL=llama3.1
# OPENAI_API_KEY=ollama

# ── Agent tuning ──
MAX_AGENT_STEPS=10
MAX_CONTEXT_TOKENS=6000
LLM_TEMPERATURE=0.1
AGENT_VERBOSE=true
```

---

## 📖 Cypher Query Examples

These queries run directly against the Neo4j Knowledge Graph. The Agentic RAG tools use similar queries internally.

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
MATCH (c:WCAGCriterion {ref_id: '1.1.1'})-[r:RELATED_CRITERION|RELATED_GUIDELINE]-(related)
RETURN c.title, type(r) AS relationship, related.ref_id, related.title;
```

### Example 4: Full LLM Context Assembly

```cypher
MATCH (c:WCAGCriterion {ref_id: '1.1.1'})
OPTIONAL MATCH (c)-[:PART_OF]->(g:WCAGGuideline)-[:PART_OF]->(p:WCAGPrinciple)
OPTIONAL MATCH (c)-[:HAS_LEVEL]->(cl:ConformanceLevel)
OPTIONAL MATCH (c)-[:HAS_SPECIAL_CASE]->(sc:WCAGSpecialCase)
OPTIONAL MATCH (c)-[:HAS_NOTE]->(n:WCAGNote)
OPTIONAL MATCH (c)-[:HAS_REFERENCE]->(r:WCAGReference)
RETURN c, g, p, cl, collect(sc) AS special_cases,
       collect(n) AS notes, collect(r) AS references;
```

### Example 5: Techniques to Pass a Criterion *(enriched)*

```cypher
MATCH (c:WCAGCriterion {ref_id: '1.4.3'})-[:HAS_TECHNIQUE]->(t:WCAGTechnique)
RETURN t.tech_id, t.title, t.technology
ORDER BY t.technology;
```

### Example 6: All Failure Patterns for Level A+AA *(enriched)*

```cypher
MATCH (c:WCAGCriterion)-[:HAS_FAILURE]->(f:WCAGTechnique)
WHERE c.level IN ['A', 'AA']
RETURN c.ref_id, c.title, collect(f.tech_id) AS failure_ids
ORDER BY c.ref_id;
```

### Example 7: Automatable Criteria with ACT Test Rules *(enriched)*

```cypher
MATCH (c:WCAGCriterion {automatable: 'full', level: 'AA'})
OPTIONAL MATCH (c)-[:HAS_TEST_RULE]->(tr:WCAGTestRule)
RETURN c.ref_id, c.title, collect(tr.title) AS act_rules;
```

### Example 8: Disability Impact Analysis *(enriched)*

```cypher
MATCH (c:WCAGCriterion)
WHERE 'cognitive' IN c.disability_impact
RETURN c.ref_id, c.title, c.level, c.wcag_version
ORDER BY c.level, c.ref_id;
```

### Example 9: Full Rule Engine Context *(enriched)*

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

### Example 10: Key Term Lookup *(enriched)*

```cypher
// Find a key term definition and which criteria use it
MATCH (c:WCAGCriterion)-[:HAS_KEY_TERM]->(kt:WCAGKeyTerm)
WHERE toLower(kt.term) CONTAINS 'programmatically determined'
RETURN kt.term, kt.definition,
       collect(DISTINCT {ref_id: c.ref_id, title: c.title}) AS used_in_criteria;
```

### Example 11: Cross-Reference — Fix Ripple Effect *(enriched)*

```cypher
// If I fix keyboard issues (2.1.1), what other criteria also benefit?
MATCH (c:WCAGCriterion {ref_id: '2.1.1'})
OPTIONAL MATCH (c)-[:RELATED_CRITERION]-(related:WCAGCriterion)
OPTIONAL MATCH (c)-[:HAS_TECHNIQUE]->(t:WCAGTechnique)<-[:HAS_TECHNIQUE]-(shared:WCAGCriterion)
WHERE shared.ref_id <> '2.1.1'
RETURN c.title AS source,
       collect(DISTINCT related.ref_id) AS directly_related,
       collect(DISTINCT shared.ref_id) AS shared_technique_siblings;
```

### Example 12: Disability Overlap Between Two Categories *(enriched)*

```cypher
// Find criteria that address both color blindness AND low vision
MATCH (c:WCAGCriterion)
WHERE any(d IN c.disability_impact WHERE toLower(d) CONTAINS 'color_blind')
  AND any(d IN c.disability_impact WHERE toLower(d) CONTAINS 'low_vision')
RETURN c.ref_id, c.title, c.level, c.disability_impact
ORDER BY c.ref_id;
```

---

## 📁 Project Structure

```
Neo4J-GraphRAG-ETL-pipeline/
│
├── 00_scrape_wcag_to_csv.py            # Step 0: Scrape W3C Understanding pages → CSV/Excel/enriched JSON
├── 01_pipeline_wcag_foundation.py      # Step 1: 5-phase ETL pipeline → Neo4j Knowledge Graph (2,418 nodes)
├── 02_agentic_rag_wcag.py             # Step 2: Agentic RAG with 8 tools + ReAct loop
│
├── wcag_22_guidelines.json             # Base WCAG 2.2 source data (87 criteria)
├── wcag_22_guidelines_enriched.json    # Enriched data (generated by Step 0, git-ignored)
├── wcag_22_all_criteria.csv            # Flat CSV export (generated by Step 0, git-ignored)
├── wcag_22_all_criteria.xlsx           # Excel export (generated by Step 0, git-ignored)
│
├── requirements.txt                    # Python dependencies
├── .env.example                        # Environment variable template
├── .env                                # Your credentials (git-ignored)
├── .gitignore                          # Git ignore rules
└── README.md                           # This file
```

---

## 🛠️ Technical Details

### ETL Constraints and Indexes

The pipeline creates:
- **Unique constraints** on `ref_id` for Principles, Guidelines, Criteria
- **Unique constraint** on `name` for ConformanceLevel
- **Unique constraint** on `tech_id` for WCAGTechnique
- **Indexes** on `code`, `title`, `level`, `wcag_version`, `automatable` for WCAGCriterion
- **Indexes** on `technology`, `category` for WCAGTechnique
- **Index** on `title` for WCAGTestRule
- **Index** on `term` for WCAGKeyTerm
- **Index** on `url` for WCAGRelatedResource

### ETL Data Integrity

- ✅ **Idempotent**: Safe to run multiple times (uses `MERGE`)
- ✅ **Referential Integrity**: All relationships validated in Phase 4
- ✅ **Batch Performance**: UNWIND-based writes (~18 transactions vs 200+ individual)
- ✅ **Auto-Detection**: Pipeline detects base vs enriched JSON automatically

### Agent Transparency

- ✅ **Full AgentTrace**: Every step logged with tool name, params, result, timing
- ✅ **Cypher Visibility**: Each tool exposes the Cypher query it ran
- ✅ **Verbose Mode**: `--verbose` flag shows real-time reasoning
- ✅ **Trace Command**: Type `trace` in interactive mode to inspect last query's steps
- ✅ **Query Plans**: Decomposed plans visible in trace with step dependencies and priorities
- ✅ **Dynamic Cypher Audit**: Every LLM-generated Cypher shown in output with guardrail status

---

## 🔮 Future Enhancements

### Phase 3: JIRA Integration
```cypher
// Link accessibility bugs to WCAG violations
CREATE (bug:JIRAIssue {key: 'ACC-123', summary: '...'})
CREATE (bug)-[:VIOLATES]->(c:WCAGCriterion {ref_id: '1.1.1'})
```

### Phase 4: REST API Server
```python
# Expose the Agentic RAG as an API
# POST /query  {"question": "What WCAG criteria apply to forms?"}
# → Agent processes query, returns structured response with citations
```

### Phase 5: Embedding Hybrid Search
```python
# Add Neo4j vector indexes for true semantic similarity
# Agent combines: graph traversal + vector search + keyword matching
# Enhanced SemanticSearchTool already searches across criteria, examples,
# benefits, and key terms — vector embeddings would further improve recall
```

### Phase 6: WCAG 3.0 Support
```python
# Extend the enrichment scraper and ETL for WCAG 3.0 when released
# Agent automatically handles multi-version queries
```

### Phase 7: Scenario Evaluation Engine
```python
# Given a URL or screenshot, decompose into UI elements (form, modal, nav)
# Then auto-invoke rule_engine per element type + cross_reference for overlaps
# Generate a full audit checklist with prioritized findings
```

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

### Agentic RAG & GraphRAG Research
- [Microsoft GraphRAG](https://microsoft.github.io/graphrag/)
- [Knowledge Graphs for RAG](https://arxiv.org/abs/2404.16130)
- [Neo4j + LLMs](https://neo4j.com/developer-blog/knowledge-graphs-llms-multi-hop-question-answering/)
- [ReAct: Synergizing Reasoning and Acting](https://arxiv.org/abs/2210.03629)
- [Tool-Augmented Language Models](https://arxiv.org/abs/2302.04761)

---

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- WCAG 2.3 / 3.0 support
- Neo4j vector index integration for hybrid search
- REST API endpoint for the Agentic RAG
- JIRA / Confluence connectors
- Scenario evaluation engine (URL/screenshot → audit checklist)
- Additional agent tools (e.g., code snippet generator, ARIA pattern recommender)
- Test coverage for all 9 agent tools + guardrail validation
- Enhanced query decomposition with parallel step execution

---

## 📄 License

This project is open source. WCAG 2.2 specification © W3C.

---

## 👤 Author

**jazxii**
- GitHub: [@jazxii](https://github.com/jazxii)

---

## 🙏 Acknowledgments

- W3C for the WCAG 2.2 specification
- Neo4j team for graph database technology
- GraphRAG and Agentic AI research community
- ReAct framework for reasoning + acting paradigm
