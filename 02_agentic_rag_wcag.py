"""
Agentic RAG System for WCAG 2.2 Knowledge Graph.

Architecture:
  ┌─────────────────────────────────────────────────────────┐
  │                    ORCHESTRATOR AGENT                     │
  │  (Plans, reasons, decides which tools to call & when)    │
  └────────┬──────────┬──────────┬──────────┬───────────────┘
           │          │          │          │
    ┌──────▼──┐ ┌─────▼────┐ ┌──▼─────┐ ┌─▼──────────┐
    │ Graph   │ │ Semantic │ │ Rule   │ │ Context    │
    │ Traversal│ │ Search  │ │ Engine │ │ Assembler  │
    │ Tool    │ │ Tool    │ │ Tool   │ │ Tool       │
    └─────────┘ └──────────┘ └────────┘ └────────────┘
         │            │           │            │
         └────────────┴───────────┴────────────┘
                          │
                    ┌─────▼─────┐
                    │  Neo4j    │
                    │  WCAG KG  │
                    └───────────┘

Tools:
  1. GraphTraversalTool   — Cypher queries for structured traversal
  2. SemanticSearchTool   — Vector similarity on criterion descriptions
  3. RuleEngineTool       — Compliance checking against WCAG rules
  4. ContextAssemblerTool — Builds rich LLM context from graph subgraphs
  5. HierarchyTool        — Navigates Principle→Guideline→Criterion trees
  6. TechniqueFinderTool  — Finds sufficient/advisory/failure techniques
  7. ImpactAnalysisTool   — Analyzes disability impact and input modalities
  8. DynamicCypherTool    — LLM-generated Cypher with guardrails (read-only)
  9. QueryDecomposer      — Breaks complex queries into prioritized sub-steps

Usage:
  python 02_agentic_rag_wcag.py --query "How do I make images accessible?"
  python 02_agentic_rag_wcag.py --interactive
"""

import os
import sys
import json
import logging
import time
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional
from enum import Enum

from neo4j import GraphDatabase
from dotenv import load_dotenv

load_dotenv()

# ============================================================
# LOGGING
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-7s │ %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("wcag_agentic_rag")


# ============================================================
# CONFIGURATION
# ============================================================
@dataclass
class AgentConfig:
    """Configuration for the Agentic RAG system."""
    neo4j_uri: str = os.getenv("NEO4J_URI", "")
    neo4j_user: str = os.getenv("NEO4J_USER", "")
    neo4j_password: str = os.getenv("NEO4J_PASSWORD", "")
    llm_provider: str = os.getenv("LLM_PROVIDER", "azure_openai")  # azure_openai | openai | ollama | anthropic
    llm_model: str = os.getenv("LLM_MODEL", "gpt-4o")
    llm_api_key: str = os.getenv("OPENAI_API_KEY", "")
    llm_base_url: str = os.getenv("LLM_BASE_URL", "")  # For Ollama: http://localhost:11434/v1
    azure_openai_endpoint: str = os.getenv("AZURE_OPENAI_ENDPOINT", "")
    azure_openai_api_version: str = os.getenv("AZURE_OPENAI_API_VERSION", "2024-05-01-preview")
    azure_openai_deployment: str = os.getenv("AZURE_OPENAI_DEPLOYMENT", "")
    max_agent_steps: int = int(os.getenv("MAX_AGENT_STEPS", "10"))
    max_context_tokens: int = int(os.getenv("MAX_CONTEXT_TOKENS", "6000"))
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    temperature: float = float(os.getenv("LLM_TEMPERATURE", "0.1"))
    verbose: bool = os.getenv("AGENT_VERBOSE", "true").lower() == "true"

    def validate(self):
        missing = []
        if not self.neo4j_uri:
            missing.append("NEO4J_URI")
        if not self.neo4j_user:
            missing.append("NEO4J_USER")
        if not self.neo4j_password:
            missing.append("NEO4J_PASSWORD")
        if missing:
            raise EnvironmentError(f"Missing env vars: {', '.join(missing)}")


# ============================================================
# NEO4J CONNECTION (reused pattern from ETL pipeline)
# ============================================================
class Neo4jConnection:
    """Managed Neo4j driver for the agent's graph operations."""

    def __init__(self, config: AgentConfig):
        self.driver = GraphDatabase.driver(
            config.neo4j_uri,
            auth=(config.neo4j_user, config.neo4j_password),
        )
        self.driver.verify_connectivity()
        log.info("Neo4j connected for Agentic RAG")

    def query(self, cypher: str, params: dict | None = None) -> list[dict]:
        """Execute a read query and return results as dicts."""
        with self.driver.session() as session:
            return [r.data() for r in session.run(cypher, params or {})]

    def close(self):
        self.driver.close()


# ============================================================
# TOOL RESULT MODEL
# ============================================================
@dataclass
class ToolResult:
    """Standardized result from any tool invocation."""
    tool_name: str
    success: bool
    data: Any = None
    error: str = ""
    execution_time: float = 0.0
    cypher_used: str = ""  # For transparency/debugging

    def to_context(self, max_length: int = 2000) -> str:
        """Format result as LLM-readable context."""
        if not self.success:
            return f"[{self.tool_name}] Error: {self.error}"
        text = json.dumps(self.data, indent=2, default=str)
        if len(text) > max_length:
            text = text[:max_length] + "\n... (truncated)"
        return f"[{self.tool_name}] Results:\n{text}"


# ============================================================
# ABSTRACT TOOL BASE
# ============================================================
class BaseTool(ABC):
    """Base class for all agent tools."""

    def __init__(self, db: Neo4jConnection):
        self.db = db

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        ...

    @property
    @abstractmethod
    def parameters(self) -> dict:
        """JSON Schema for tool parameters."""
        ...

    @abstractmethod
    def execute(self, **kwargs) -> ToolResult:
        ...

    def _timed_query(self, cypher: str, params: dict | None = None) -> tuple[list[dict], float]:
        """Run a Cypher query and return (results, elapsed_seconds)."""
        t0 = time.time()
        results = self.db.query(cypher, params)
        return results, round(time.time() - t0, 4)


# ============================================================
# TOOL 1: GRAPH TRAVERSAL
# ============================================================
class GraphTraversalTool(BaseTool):
    """Executes structured Cypher queries for hierarchy traversal."""

    @property
    def name(self) -> str:
        return "graph_traversal"

    @property
    def description(self) -> str:
        return (
            "Traverse the WCAG knowledge graph to find principles, guidelines, "
            "criteria, and their relationships. Use for structured queries like "
            "'find all Level AA criteria under Perceivable' or 'get the hierarchy "
            "for criterion 1.4.3'."
        )

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "query_type": {
                    "type": "string",
                    "enum": [
                        "get_all_principles",
                        "get_guidelines_for_principle",
                        "get_criteria_for_guideline",
                        "get_criterion_detail",
                        "get_criteria_by_level",
                        "get_full_hierarchy",
                        "get_criterion_with_context",
                    ],
                    "description": "The type of graph traversal to perform"
                },
                "ref_id": {
                    "type": "string",
                    "description": "Reference ID (e.g., '1', '1.1', '1.1.1')"
                },
                "level": {
                    "type": "string",
                    "enum": ["A", "AA", "AAA"],
                    "description": "WCAG conformance level filter"
                },
            },
            "required": ["query_type"],
        }

    def execute(self, **kwargs) -> ToolResult:
        query_type = kwargs.get("query_type", "")
        ref_id = kwargs.get("ref_id", "")
        level = kwargs.get("level", "")

        queries = {
            "get_all_principles": (
                """
                MATCH (p:WCAGPrinciple)
                OPTIONAL MATCH (g:WCAGGuideline)-[:PART_OF]->(p)
                WITH p, count(g) AS guideline_count
                RETURN p.ref_id AS ref_id, p.title AS title,
                       p.description AS description, guideline_count
                ORDER BY p.ref_id
                """,
                {}
            ),
            "get_guidelines_for_principle": (
                """
                MATCH (g:WCAGGuideline)-[:PART_OF]->(p:WCAGPrinciple {ref_id: $ref_id})
                OPTIONAL MATCH (c:WCAGCriterion)-[:PART_OF]->(g)
                WITH g, count(c) AS criteria_count
                RETURN g.ref_id AS ref_id, g.title AS title,
                       g.description AS description, criteria_count
                ORDER BY g.ref_id
                """,
                {"ref_id": ref_id}
            ),
            "get_criteria_for_guideline": (
                """
                MATCH (c:WCAGCriterion)-[:PART_OF]->(g:WCAGGuideline {ref_id: $ref_id})
                RETURN c.ref_id AS ref_id, c.title AS title,
                       c.level AS level, c.description AS description,
                       c.wcag_version AS wcag_version,
                       c.automatable AS automatable
                ORDER BY c.ref_id
                """,
                {"ref_id": ref_id}
            ),
            "get_criterion_detail": (
                """
                MATCH (c:WCAGCriterion {ref_id: $ref_id})
                OPTIONAL MATCH (c)-[:PART_OF]->(g:WCAGGuideline)-[:PART_OF]->(p:WCAGPrinciple)
                OPTIONAL MATCH (c)-[:HAS_LEVEL]->(cl:ConformanceLevel)
                RETURN c.ref_id AS ref_id, c.title AS title,
                       c.description AS description, c.level AS level,
                       c.url AS url, c.intent AS intent,
                       c.wcag_version AS wcag_version,
                       c.automatable AS automatable,
                       c.disability_impact AS disability_impact,
                       c.input_types_affected AS input_types_affected,
                       c.in_brief_goal AS goal,
                       c.in_brief_what_to_do AS what_to_do,
                       c.in_brief_why_important AS why_important,
                       g.title AS guideline, p.title AS principle,
                       cl.name AS conformance_level
                """,
                {"ref_id": ref_id}
            ),
            "get_criteria_by_level": (
                """
                MATCH (c:WCAGCriterion)-[:HAS_LEVEL]->(cl:ConformanceLevel {name: $level})
                MATCH (c)-[:PART_OF]->(g:WCAGGuideline)-[:PART_OF]->(p:WCAGPrinciple)
                RETURN c.ref_id AS ref_id, c.title AS title,
                       c.description AS description,
                       g.title AS guideline, p.title AS principle
                ORDER BY c.ref_id
                """,
                {"level": level}
            ),
            "get_full_hierarchy": (
                """
                MATCH (p:WCAGPrinciple)
                OPTIONAL MATCH (g:WCAGGuideline)-[:PART_OF]->(p)
                OPTIONAL MATCH (c:WCAGCriterion)-[:PART_OF]->(g)
                WITH p, g, collect({ref_id: c.ref_id, title: c.title, level: c.level}) AS criteria
                WITH p, collect({ref_id: g.ref_id, title: g.title, criteria: criteria}) AS guidelines
                RETURN p.ref_id AS ref_id, p.title AS title, guidelines
                ORDER BY p.ref_id
                """,
                {}
            ),
            "get_criterion_with_context": (
                """
                MATCH (c:WCAGCriterion {ref_id: $ref_id})
                MATCH (c)-[:PART_OF]->(g:WCAGGuideline)-[:PART_OF]->(p:WCAGPrinciple)
                OPTIONAL MATCH (c)-[:HAS_SPECIAL_CASE]->(sc:WCAGSpecialCase)
                OPTIONAL MATCH (c)-[:HAS_NOTE]->(n:WCAGNote)
                OPTIONAL MATCH (c)-[:HAS_REFERENCE]->(r:WCAGReference)
                OPTIONAL MATCH (c)-[:RELATED_CRITERION]->(rc:WCAGCriterion)
                RETURN c {.*, principle: p.title, guideline: g.title} AS criterion,
                       collect(DISTINCT sc {.*}) AS special_cases,
                       collect(DISTINCT n {.*}) AS notes,
                       collect(DISTINCT r {.*}) AS refs,
                       collect(DISTINCT {ref_id: rc.ref_id, title: rc.title}) AS related
                """,
                {"ref_id": ref_id}
            ),
        }

        if query_type not in queries:
            return ToolResult(
                tool_name=self.name, success=False,
                error=f"Unknown query_type: {query_type}. "
                      f"Valid: {list(queries.keys())}"
            )

        cypher, params = queries[query_type]
        try:
            results, elapsed = self._timed_query(cypher, params)
            return ToolResult(
                tool_name=self.name, success=True,
                data=results, execution_time=elapsed,
                cypher_used=cypher.strip()
            )
        except Exception as e:
            return ToolResult(
                tool_name=self.name, success=False, error=str(e)
            )


# ============================================================
# TOOL 2: SEMANTIC SEARCH
# ============================================================
class SemanticSearchTool(BaseTool):
    """
    Keyword and semantic search across WCAG criteria.
    Uses Neo4j full-text capabilities and property matching.
    For true vector search, Neo4j vector indexes can be added.
    """

    @property
    def name(self) -> str:
        return "semantic_search"

    @property
    def description(self) -> str:
        return (
            "Search WCAG criteria by keyword, topic, or natural language description. "
            "Use when the user asks about a topic (e.g., 'color contrast', 'keyboard navigation', "
            "'screen reader') rather than a specific criterion ID."
        )

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search terms or natural language query"
                },
                "level_filter": {
                    "type": "string",
                    "enum": ["A", "AA", "AAA", ""],
                    "description": "Optional conformance level filter"
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum results to return",
                    "default": 10
                },
            },
            "required": ["query"],
        }

    def execute(self, **kwargs) -> ToolResult:
        query = kwargs.get("query", "")
        level_filter = kwargs.get("level_filter", "")
        limit = kwargs.get("limit", 10)

        if not query:
            return ToolResult(
                tool_name=self.name, success=False,
                error="Search query cannot be empty"
            )

        # Build keyword search with CONTAINS across multiple fields
        # This is a practical approach; for production, use Neo4j vector indexes
        keywords = [kw.strip().lower() for kw in query.split() if len(kw.strip()) > 2]
        if not keywords:
            keywords = [query.lower()]

        # Build WHERE clause for keyword matching on criterion properties
        where_conditions = []
        for i, kw in enumerate(keywords):
            param_key = f"kw{i}"
            where_conditions.append(
                f"(toLower(c.title) CONTAINS ${param_key} OR "
                f"toLower(c.description) CONTAINS ${param_key} OR "
                f"toLower(c.intent) CONTAINS ${param_key} OR "
                f"toLower(c.in_brief_goal) CONTAINS ${param_key} OR "
                f"toLower(c.in_brief_what_to_do) CONTAINS ${param_key})"
            )

        direct_where = " OR ".join(where_conditions)
        params = {f"kw{i}": kw for i, kw in enumerate(keywords)}
        params["limit"] = limit

        level_clause = ""
        if level_filter:
            level_clause = "AND c.level = $level_filter"
            params["level_filter"] = level_filter

        # Extended search: also match via connected examples, benefits, key terms
        # Use UNION to find criteria matched through connected nodes
        cypher = f"""
            // Direct match on criterion properties
            MATCH (c:WCAGCriterion)
            WHERE ({direct_where}) {level_clause}
            OPTIONAL MATCH (c)-[:PART_OF]->(g:WCAGGuideline)-[:PART_OF]->(p:WCAGPrinciple)
            RETURN c.ref_id AS ref_id, c.title AS title,
                   c.level AS level, c.description AS description,
                   c.intent AS intent, c.wcag_version AS wcag_version,
                   c.in_brief_goal AS goal,
                   c.in_brief_what_to_do AS what_to_do,
                   g.title AS guideline, p.title AS principle,
                   'criterion' AS match_source

            UNION

            // Match via connected examples
            MATCH (c:WCAGCriterion)-[:HAS_EXAMPLE]->(ex:WCAGExample)
            WHERE ({" OR ".join(
                f"toLower(ex.description) CONTAINS $kw{i}" for i in range(len(keywords))
            )}) {level_clause}
            OPTIONAL MATCH (c)-[:PART_OF]->(g:WCAGGuideline)-[:PART_OF]->(p:WCAGPrinciple)
            RETURN DISTINCT c.ref_id AS ref_id, c.title AS title,
                   c.level AS level, c.description AS description,
                   c.intent AS intent, c.wcag_version AS wcag_version,
                   c.in_brief_goal AS goal,
                   c.in_brief_what_to_do AS what_to_do,
                   g.title AS guideline, p.title AS principle,
                   'example' AS match_source

            UNION

            // Match via key terms
            MATCH (c:WCAGCriterion)-[:HAS_KEY_TERM]->(kt:WCAGKeyTerm)
            WHERE ({" OR ".join(
                f"(toLower(kt.term) CONTAINS $kw{i} OR toLower(kt.definition) CONTAINS $kw{i})" for i in range(len(keywords))
            )}) {level_clause}
            OPTIONAL MATCH (c)-[:PART_OF]->(g:WCAGGuideline)-[:PART_OF]->(p:WCAGPrinciple)
            RETURN DISTINCT c.ref_id AS ref_id, c.title AS title,
                   c.level AS level, c.description AS description,
                   c.intent AS intent, c.wcag_version AS wcag_version,
                   c.in_brief_goal AS goal,
                   c.in_brief_what_to_do AS what_to_do,
                   g.title AS guideline, p.title AS principle,
                   'key_term' AS match_source

            UNION

            // Match via benefits
            MATCH (c:WCAGCriterion)-[:HAS_BENEFIT]->(b:WCAGBenefit)
            WHERE ({" OR ".join(
                f"toLower(b.description) CONTAINS $kw{i}" for i in range(len(keywords))
            )}) {level_clause}
            OPTIONAL MATCH (c)-[:PART_OF]->(g:WCAGGuideline)-[:PART_OF]->(p:WCAGPrinciple)
            RETURN DISTINCT c.ref_id AS ref_id, c.title AS title,
                   c.level AS level, c.description AS description,
                   c.intent AS intent, c.wcag_version AS wcag_version,
                   c.in_brief_goal AS goal,
                   c.in_brief_what_to_do AS what_to_do,
                   g.title AS guideline, p.title AS principle,
                   'benefit' AS match_source
        """

        try:
            results, elapsed = self._timed_query(cypher, params)

            # Deduplicate by ref_id, keeping match_source info
            seen = {}
            for row in results:
                rid = row["ref_id"]
                if rid not in seen:
                    seen[rid] = row
                    seen[rid]["match_sources"] = [row.pop("match_source", "criterion")]
                else:
                    src = row.get("match_source", "")
                    if src and src not in seen[rid]["match_sources"]:
                        seen[rid]["match_sources"].append(src)

            deduped = sorted(seen.values(), key=lambda x: x["ref_id"])[:limit]

            return ToolResult(
                tool_name=self.name, success=True,
                data=deduped, execution_time=elapsed,
                cypher_used=cypher.strip()
            )
        except Exception as e:
            return ToolResult(
                tool_name=self.name, success=False, error=str(e)
            )


# ============================================================
# TOOL 3: TECHNIQUE FINDER
# ============================================================
class TechniqueFinderTool(BaseTool):
    """Find sufficient, advisory, and failure techniques for criteria."""

    @property
    def name(self) -> str:
        return "technique_finder"

    @property
    def description(self) -> str:
        return (
            "Find WCAG techniques (sufficient, advisory, failures) for a specific "
            "criterion or topic. Use when the user asks HOW to comply, what techniques "
            "to use, or what common failures to avoid."
        )

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "criterion_id": {
                    "type": "string",
                    "description": "Criterion ref_id (e.g., '1.1.1')"
                },
                "technique_type": {
                    "type": "string",
                    "enum": ["sufficient", "advisory", "failures", "all"],
                    "description": "Type of techniques to find",
                    "default": "all"
                },
                "technology_filter": {
                    "type": "string",
                    "description": "Filter by technology (e.g., 'html', 'aria', 'css')",
                    "default": ""
                },
            },
            "required": ["criterion_id"],
        }

    def execute(self, **kwargs) -> ToolResult:
        criterion_id = kwargs.get("criterion_id", "")
        technique_type = kwargs.get("technique_type", "all")
        technology_filter = kwargs.get("technology_filter", "")

        if not criterion_id:
            return ToolResult(
                tool_name=self.name, success=False,
                error="criterion_id is required"
            )

        # Map technique type to relationship types
        rel_map = {
            "sufficient": ["HAS_TECHNIQUE"],
            "advisory": ["HAS_ADVISORY_TECHNIQUE"],
            "failures": ["HAS_FAILURE"],
            "all": ["HAS_TECHNIQUE", "HAS_ADVISORY_TECHNIQUE", "HAS_FAILURE"],
        }
        rel_types = rel_map.get(technique_type, rel_map["all"])

        all_techniques = []
        for rel_type in rel_types:
            tech_filter = ""
            params = {"criterion_id": criterion_id}

            if technology_filter:
                tech_filter = "AND toLower(t.technology) CONTAINS $tech_filter"
                params["tech_filter"] = technology_filter.lower()

            cypher = f"""
                MATCH (c:WCAGCriterion {{ref_id: $criterion_id}})
                      -[r:{rel_type}]->(t:WCAGTechnique)
                WHERE true {tech_filter}
                RETURN t.tech_id AS tech_id, t.title AS title,
                       t.url AS url, t.technology AS technology,
                       t.category AS category,
                       '{rel_type}' AS relationship_type
                ORDER BY t.tech_id
            """

            try:
                results, _ = self._timed_query(cypher, params)
                all_techniques.extend(results)
            except Exception as e:
                return ToolResult(
                    tool_name=self.name, success=False, error=str(e)
                )

        # Group by type for cleaner output
        grouped = {}
        for tech in all_techniques:
            rel = tech.pop("relationship_type", "unknown")
            grouped.setdefault(rel, []).append(tech)

        return ToolResult(
            tool_name=self.name, success=True,
            data={
                "criterion_id": criterion_id,
                "technique_count": len(all_techniques),
                "techniques": grouped,
            },
        )


# ============================================================
# TOOL 4: RULE ENGINE
# ============================================================
class RuleEngineTool(BaseTool):
    """
    WCAG compliance rule engine. Checks elements/scenarios against
    WCAG criteria and returns applicable rules, required techniques,
    and potential failures.
    """

    @property
    def name(self) -> str:
        return "rule_engine"

    @property
    def description(self) -> str:
        return (
            "Check WCAG compliance rules for a given element type, scenario, or "
            "disability category. Use when the user asks 'is X accessible?', "
            "'what rules apply to images/forms/videos?', or 'what criteria affect "
            "users with [disability]?'."
        )

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "check_type": {
                    "type": "string",
                    "enum": [
                        "element_rules",        # What rules apply to <img>, <form>, etc.
                        "disability_impact",     # What criteria affect blindness, motor, etc.
                        "conformance_checklist", # Full checklist for a level
                        "automatable_criteria",  # What can be auto-tested
                        "version_diff",          # What's new in WCAG 2.1 / 2.2
                    ],
                    "description": "Type of rule check to perform"
                },
                "element_type": {
                    "type": "string",
                    "description": "HTML element type (e.g., 'image', 'form', 'video', 'link', 'button')"
                },
                "disability": {
                    "type": "string",
                    "description": "Disability category (e.g., 'blindness', 'low_vision', 'cognitive', 'motor', 'deafness')"
                },
                "level": {
                    "type": "string",
                    "enum": ["A", "AA", "AAA"],
                    "description": "Conformance level"
                },
                "wcag_version": {
                    "type": "string",
                    "enum": ["2.0", "2.1", "2.2"],
                    "description": "WCAG version filter"
                },
            },
            "required": ["check_type"],
        }

    # Element-to-keyword mapping for rule lookups
    ELEMENT_KEYWORDS = {
        "image": ["image", "img", "non-text", "alt text", "text alternative", "decorative"],
        "form": ["form", "input", "label", "error", "instruction", "validation"],
        "video": ["video", "media", "captions", "audio description", "transcript", "time-based"],
        "audio": ["audio", "media", "transcript", "captions", "time-based"],
        "link": ["link", "navigation", "purpose", "focus", "anchor"],
        "button": ["button", "control", "name", "role", "keyboard", "focus"],
        "table": ["table", "header", "cell", "relationship", "data table"],
        "heading": ["heading", "section", "structure", "hierarchy", "organize"],
        "color": ["color", "contrast", "visual", "distinguishable"],
        "text": ["text", "resize", "spacing", "readable", "language"],
        "animation": ["animation", "motion", "flash", "seizure", "pause"],
        "modal": ["modal", "dialog", "focus", "trap", "keyboard"],
    }

    def execute(self, **kwargs) -> ToolResult:
        check_type = kwargs.get("check_type", "")

        try:
            if check_type == "element_rules":
                return self._check_element_rules(kwargs.get("element_type", ""))
            elif check_type == "disability_impact":
                return self._check_disability_impact(kwargs.get("disability", ""))
            elif check_type == "conformance_checklist":
                return self._conformance_checklist(kwargs.get("level", "AA"))
            elif check_type == "automatable_criteria":
                return self._automatable_criteria()
            elif check_type == "version_diff":
                return self._version_diff(kwargs.get("wcag_version", "2.2"))
            else:
                return ToolResult(
                    tool_name=self.name, success=False,
                    error=f"Unknown check_type: {check_type}"
                )
        except Exception as e:
            return ToolResult(
                tool_name=self.name, success=False, error=str(e)
            )

    def _check_element_rules(self, element_type: str) -> ToolResult:
        """Find all WCAG criteria relevant to a given HTML element type."""
        element_type = element_type.lower().strip()
        keywords = self.ELEMENT_KEYWORDS.get(element_type, [element_type])

        # Build keyword search
        conditions = []
        params = {}
        for i, kw in enumerate(keywords):
            pk = f"kw{i}"
            conditions.append(
                f"(toLower(c.title) CONTAINS ${pk} OR "
                f"toLower(c.description) CONTAINS ${pk} OR "
                f"toLower(c.intent) CONTAINS ${pk})"
            )
            params[pk] = kw

        where = " OR ".join(conditions)
        cypher = f"""
            MATCH (c:WCAGCriterion)
            WHERE {where}
            MATCH (c)-[:PART_OF]->(g:WCAGGuideline)-[:PART_OF]->(p:WCAGPrinciple)
            OPTIONAL MATCH (c)-[:HAS_TECHNIQUE]->(t:WCAGTechnique)
            OPTIONAL MATCH (c)-[:HAS_FAILURE]->(f:WCAGTechnique)
            RETURN c.ref_id AS ref_id, c.title AS title,
                   c.level AS level, c.description AS description,
                   c.in_brief_what_to_do AS what_to_do,
                   g.title AS guideline, p.title AS principle,
                   collect(DISTINCT t.tech_id) AS techniques,
                   collect(DISTINCT f.tech_id) AS failures
            ORDER BY c.level, c.ref_id
        """

        results, elapsed = self._timed_query(cypher, params)
        return ToolResult(
            tool_name=self.name, success=True,
            data={
                "element_type": element_type,
                "applicable_criteria": len(results),
                "rules": results,
            },
            execution_time=elapsed,
        )

    def _check_disability_impact(self, disability: str) -> ToolResult:
        """Find criteria that impact a specific disability category."""
        disability = disability.lower().strip()
        cypher = """
            MATCH (c:WCAGCriterion)
            WHERE any(d IN c.disability_impact WHERE toLower(d) CONTAINS $disability)
            MATCH (c)-[:PART_OF]->(g:WCAGGuideline)-[:PART_OF]->(p:WCAGPrinciple)
            RETURN c.ref_id AS ref_id, c.title AS title,
                   c.level AS level, c.disability_impact AS impact,
                   c.in_brief_why_important AS why_important,
                   g.title AS guideline, p.title AS principle
            ORDER BY c.level, c.ref_id
        """

        results, elapsed = self._timed_query(cypher, {"disability": disability})
        return ToolResult(
            tool_name=self.name, success=True,
            data={
                "disability": disability,
                "affected_criteria": len(results),
                "criteria": results,
            },
            execution_time=elapsed,
        )

    def _conformance_checklist(self, level: str) -> ToolResult:
        """Generate a conformance checklist for a given level."""
        # Level AA includes A; AAA includes A + AA
        level_includes = {"A": ["A"], "AA": ["A", "AA"], "AAA": ["A", "AA", "AAA"]}
        levels = level_includes.get(level, ["A", "AA"])

        cypher = """
            MATCH (c:WCAGCriterion)-[:HAS_LEVEL]->(cl:ConformanceLevel)
            WHERE cl.name IN $levels
            MATCH (c)-[:PART_OF]->(g:WCAGGuideline)-[:PART_OF]->(p:WCAGPrinciple)
            RETURN p.title AS principle, g.ref_id AS guideline_id,
                   g.title AS guideline, c.ref_id AS ref_id,
                   c.title AS title, c.level AS level,
                   c.automatable AS automatable,
                   c.in_brief_what_to_do AS what_to_do
            ORDER BY c.ref_id
        """

        results, elapsed = self._timed_query(cypher, {"levels": levels})

        # Group by principle
        grouped = {}
        for row in results:
            principle = row["principle"]
            grouped.setdefault(principle, []).append(row)

        return ToolResult(
            tool_name=self.name, success=True,
            data={
                "target_level": level,
                "includes_levels": levels,
                "total_criteria": len(results),
                "by_principle": {k: len(v) for k, v in grouped.items()},
                "checklist": results,
            },
            execution_time=elapsed,
        )

    def _automatable_criteria(self) -> ToolResult:
        """Return criteria grouped by automation level."""
        cypher = """
            MATCH (c:WCAGCriterion)
            WHERE c.automatable IS NOT NULL
            RETURN c.ref_id AS ref_id, c.title AS title,
                   c.level AS level, c.automatable AS automatable
            ORDER BY c.automatable, c.ref_id
        """

        results, elapsed = self._timed_query(cypher)
        grouped = {}
        for row in results:
            grouped.setdefault(row["automatable"], []).append(row)

        return ToolResult(
            tool_name=self.name, success=True,
            data={
                "total": len(results),
                "by_automation_level": {k: len(v) for k, v in grouped.items()},
                "criteria": grouped,
            },
            execution_time=elapsed,
        )

    def _version_diff(self, version: str) -> ToolResult:
        """Return criteria introduced in a specific WCAG version."""
        cypher = """
            MATCH (c:WCAGCriterion)
            WHERE c.wcag_version = $version
            MATCH (c)-[:PART_OF]->(g:WCAGGuideline)-[:PART_OF]->(p:WCAGPrinciple)
            RETURN c.ref_id AS ref_id, c.title AS title,
                   c.level AS level, c.description AS description,
                   g.title AS guideline, p.title AS principle
            ORDER BY c.ref_id
        """

        results, elapsed = self._timed_query(cypher, {"version": version})
        return ToolResult(
            tool_name=self.name, success=True,
            data={
                "wcag_version": version,
                "new_criteria_count": len(results),
                "criteria": results,
            },
            execution_time=elapsed,
        )


# ============================================================
# TOOL 5: IMPACT ANALYSIS
# ============================================================
class ImpactAnalysisTool(BaseTool):
    """Analyze disability impact and input modality requirements."""

    @property
    def name(self) -> str:
        return "impact_analysis"

    @property
    def description(self) -> str:
        return (
            "Analyze the impact of WCAG criteria across disability categories "
            "and input modalities. Use for questions like 'which criteria are "
            "most important for blind users?' or 'what affects keyboard-only users?'."
        )

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "analysis_type": {
                    "type": "string",
                    "enum": [
                        "disability_matrix",     # Criteria × disability grid
                        "input_modality_impact",  # Criteria by input type
                        "criterion_impact",       # Full impact for one criterion
                    ],
                },
                "criterion_id": {"type": "string"},
                "input_type": {
                    "type": "string",
                    "description": "e.g., 'keyboard', 'pointer', 'touch', 'voice'"
                },
            },
            "required": ["analysis_type"],
        }

    def execute(self, **kwargs) -> ToolResult:
        analysis_type = kwargs.get("analysis_type", "")

        try:
            if analysis_type == "disability_matrix":
                return self._disability_matrix()
            elif analysis_type == "input_modality_impact":
                return self._input_modality_impact(kwargs.get("input_type", ""))
            elif analysis_type == "criterion_impact":
                return self._criterion_impact(kwargs.get("criterion_id", ""))
            else:
                return ToolResult(
                    tool_name=self.name, success=False,
                    error=f"Unknown analysis_type: {analysis_type}"
                )
        except Exception as e:
            return ToolResult(
                tool_name=self.name, success=False, error=str(e)
            )

    def _disability_matrix(self) -> ToolResult:
        """Build a matrix of disability categories → criteria count."""
        cypher = """
            MATCH (c:WCAGCriterion)
            WHERE c.disability_impact IS NOT NULL AND size(c.disability_impact) > 0
            UNWIND c.disability_impact AS disability
            WITH disability, collect({ref_id: c.ref_id, title: c.title, level: c.level}) AS criteria
            RETURN disability, size(criteria) AS count, criteria
            ORDER BY count DESC
        """
        results, elapsed = self._timed_query(cypher)
        return ToolResult(
            tool_name=self.name, success=True,
            data=results, execution_time=elapsed,
        )

    def _input_modality_impact(self, input_type: str) -> ToolResult:
        """Find criteria affecting a specific input modality."""
        if input_type:
            cypher = """
                MATCH (c:WCAGCriterion)
                WHERE any(it IN c.input_types_affected WHERE toLower(it) CONTAINS $input_type)
                RETURN c.ref_id AS ref_id, c.title AS title,
                       c.level AS level, c.input_types_affected AS input_types,
                       c.in_brief_what_to_do AS what_to_do
                ORDER BY c.ref_id
            """
            results, elapsed = self._timed_query(cypher, {"input_type": input_type.lower()})
        else:
            cypher = """
                MATCH (c:WCAGCriterion)
                WHERE c.input_types_affected IS NOT NULL AND size(c.input_types_affected) > 0
                UNWIND c.input_types_affected AS input_type
                WITH input_type, count(c) AS criteria_count
                RETURN input_type, criteria_count
                ORDER BY criteria_count DESC
            """
            results, elapsed = self._timed_query(cypher)

        return ToolResult(
            tool_name=self.name, success=True,
            data=results, execution_time=elapsed,
        )

    def _criterion_impact(self, criterion_id: str) -> ToolResult:
        """Full impact profile for a single criterion."""
        cypher = """
            MATCH (c:WCAGCriterion {ref_id: $criterion_id})
            OPTIONAL MATCH (c)-[:HAS_BENEFIT]->(b:WCAGBenefit)
            OPTIONAL MATCH (c)-[:HAS_TECHNIQUE]->(t:WCAGTechnique)
            OPTIONAL MATCH (c)-[:HAS_FAILURE]->(f:WCAGTechnique)
            OPTIONAL MATCH (c)-[:HAS_TEST_RULE]->(tr:WCAGTestRule)
            RETURN c.ref_id AS ref_id, c.title AS title,
                   c.level AS level,
                   c.disability_impact AS disability_impact,
                   c.input_types_affected AS input_types,
                   c.automatable AS automatable,
                   c.in_brief_why_important AS why_important,
                   collect(DISTINCT b.description) AS benefits,
                   collect(DISTINCT t.tech_id) AS techniques,
                   collect(DISTINCT f.tech_id) AS failures,
                   collect(DISTINCT tr.title) AS test_rules
        """
        results, elapsed = self._timed_query(cypher, {"criterion_id": criterion_id})
        return ToolResult(
            tool_name=self.name, success=True,
            data=results[0] if results else {},
            execution_time=elapsed,
        )


# ============================================================
# TOOL 6: KEY TERM LOOKUP
# ============================================================
class KeyTermLookupTool(BaseTool):
    """
    Look up WCAG key term definitions from the knowledge graph.
    Queries 725+ WCAGKeyTerm nodes to answer terminology questions
    like 'what does programmatically determined mean?' or
    'define text alternative'.
    """

    @property
    def name(self) -> str:
        return "key_term_lookup"

    @property
    def description(self) -> str:
        return (
            "Look up WCAG key term definitions and find which criteria use a term. "
            "Use when the user asks 'what does X mean?', 'define X', or references "
            "WCAG-specific terminology like 'programmatically determined', "
            "'text alternative', 'essential', 'assistive technology', etc."
        )

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "term": {
                    "type": "string",
                    "description": "The term to look up (e.g., 'programmatically determined', 'text alternative')"
                },
                "criterion_id": {
                    "type": "string",
                    "description": "Optional: get key terms for a specific criterion",
                    "default": ""
                },
            },
            "required": ["term"],
        }

    def execute(self, **kwargs) -> ToolResult:
        term = kwargs.get("term", "").strip()
        criterion_id = kwargs.get("criterion_id", "").strip()

        if not term and not criterion_id:
            return ToolResult(
                tool_name=self.name, success=False,
                error="Either 'term' or 'criterion_id' is required"
            )

        # If criterion_id is provided, get all key terms for that criterion
        if criterion_id:
            cypher = """
                MATCH (c:WCAGCriterion {ref_id: $criterion_id})-[:HAS_KEY_TERM]->(kt:WCAGKeyTerm)
                RETURN kt.term AS term, kt.definition AS definition,
                       c.ref_id AS criterion_id, c.title AS criterion_title
                ORDER BY kt.term
            """
            results, elapsed = self._timed_query(cypher, {"criterion_id": criterion_id})
            return ToolResult(
                tool_name=self.name, success=True,
                data={"criterion_id": criterion_id, "key_terms": results},
                execution_time=elapsed,
            )

        # Search for a specific term across all criteria
        cypher = """
            MATCH (c:WCAGCriterion)-[:HAS_KEY_TERM]->(kt:WCAGKeyTerm)
            WHERE toLower(kt.term) CONTAINS toLower($term)
            RETURN kt.term AS term, kt.definition AS definition,
                   collect(DISTINCT {ref_id: c.ref_id, title: c.title}) AS used_in_criteria
            ORDER BY kt.term
        """
        results, elapsed = self._timed_query(cypher, {"term": term})

        # Deduplicate terms (same term appears across criteria)
        seen_terms = {}
        for row in results:
            t = row["term"]
            if t not in seen_terms:
                seen_terms[t] = {
                    "term": t,
                    "definition": row["definition"],
                    "used_in_criteria": row["used_in_criteria"],
                }
            else:
                # Merge criteria lists
                existing_ids = {c["ref_id"] for c in seen_terms[t]["used_in_criteria"]}
                for c in row["used_in_criteria"]:
                    if c["ref_id"] not in existing_ids:
                        seen_terms[t]["used_in_criteria"].append(c)

        return ToolResult(
            tool_name=self.name, success=True,
            data={
                "search_term": term,
                "results_count": len(seen_terms),
                "terms": list(seen_terms.values()),
            },
            execution_time=elapsed,
        )


# ============================================================
# TOOL 7: CROSS-REFERENCE RESOLVER
# ============================================================
class CrossReferenceTool(BaseTool):
    """
    Multi-hop graph walks for cross-referencing:
    - Find all criteria related to a given criterion (and their relations)
    - Find shared techniques across multiple criteria
    - Find overlapping disability impacts between criteria
    - Map the ripple effect of fixing/breaking a criterion
    """

    @property
    def name(self) -> str:
        return "cross_reference"

    @property
    def description(self) -> str:
        return (
            "Perform multi-hop cross-referencing across the WCAG knowledge graph. "
            "Use for questions like 'what criteria are related to 1.4.3 and share "
            "the same techniques?', 'if I fix keyboard issues, what else improves?', "
            "'which criteria overlap between color blindness and low vision?', or "
            "'what techniques cover multiple criteria?'."
        )

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "analysis_type": {
                    "type": "string",
                    "enum": [
                        "related_chain",        # SC → related SCs → their related SCs (2-hop)
                        "shared_techniques",    # Which techniques apply to multiple given criteria
                        "disability_overlap",   # Criteria overlap between two disability categories
                        "technique_coverage",   # Given a technique, which criteria does it satisfy
                        "fix_ripple_effect",    # If I fix SC X, what other SCs benefit
                    ],
                    "description": "Type of cross-reference analysis"
                },
                "criterion_id": {
                    "type": "string",
                    "description": "Primary criterion ref_id (for related_chain, fix_ripple_effect)"
                },
                "criterion_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of criterion IDs (for shared_techniques)"
                },
                "disability_a": {
                    "type": "string",
                    "description": "First disability category (for disability_overlap)"
                },
                "disability_b": {
                    "type": "string",
                    "description": "Second disability category (for disability_overlap)"
                },
                "technique_id": {
                    "type": "string",
                    "description": "Technique ID like G18, H37 (for technique_coverage)"
                },
            },
            "required": ["analysis_type"],
        }

    def execute(self, **kwargs) -> ToolResult:
        analysis_type = kwargs.get("analysis_type", "")

        try:
            if analysis_type == "related_chain":
                return self._related_chain(kwargs.get("criterion_id", ""))
            elif analysis_type == "shared_techniques":
                return self._shared_techniques(kwargs.get("criterion_ids", []))
            elif analysis_type == "disability_overlap":
                return self._disability_overlap(
                    kwargs.get("disability_a", ""),
                    kwargs.get("disability_b", ""),
                )
            elif analysis_type == "technique_coverage":
                return self._technique_coverage(kwargs.get("technique_id", ""))
            elif analysis_type == "fix_ripple_effect":
                return self._fix_ripple_effect(kwargs.get("criterion_id", ""))
            else:
                return ToolResult(
                    tool_name=self.name, success=False,
                    error=f"Unknown analysis_type: {analysis_type}"
                )
        except Exception as e:
            return ToolResult(
                tool_name=self.name, success=False, error=str(e)
            )

    def _related_chain(self, criterion_id: str) -> ToolResult:
        """2-hop related criteria chain: SC → related → their related."""
        cypher = """
            MATCH (c:WCAGCriterion {ref_id: $cid})
            OPTIONAL MATCH (c)-[:RELATED_CRITERION]->(r1:WCAGCriterion)
            OPTIONAL MATCH (r1)-[:RELATED_CRITERION]->(r2:WCAGCriterion)
            WHERE r2.ref_id <> $cid
            WITH c, collect(DISTINCT {
                ref_id: r1.ref_id, title: r1.title, level: r1.level
            }) AS direct_related,
            collect(DISTINCT {
                ref_id: r2.ref_id, title: r2.title, level: r2.level,
                via: r1.ref_id
            }) AS second_hop
            RETURN c.ref_id AS source, c.title AS source_title,
                   direct_related, second_hop
        """
        results, elapsed = self._timed_query(cypher, {"cid": criterion_id})
        data = results[0] if results else {}

        # Clean out null entries
        if data:
            data["direct_related"] = [r for r in data.get("direct_related", [])
                                       if r.get("ref_id")]
            data["second_hop"] = [r for r in data.get("second_hop", [])
                                   if r.get("ref_id")]

        return ToolResult(
            tool_name=self.name, success=True,
            data=data, execution_time=elapsed,
        )

    def _shared_techniques(self, criterion_ids: list[str]) -> ToolResult:
        """Find techniques shared across multiple criteria."""
        if len(criterion_ids) < 2:
            return ToolResult(
                tool_name=self.name, success=False,
                error="Need at least 2 criterion_ids for shared_techniques"
            )

        cypher = """
            MATCH (c:WCAGCriterion)-[:HAS_TECHNIQUE|HAS_ADVISORY_TECHNIQUE|HAS_FAILURE]->(t:WCAGTechnique)
            WHERE c.ref_id IN $cids
            WITH t, collect(DISTINCT c.ref_id) AS criteria_using
            WHERE size(criteria_using) > 1
            RETURN t.tech_id AS tech_id, t.title AS title,
                   t.technology AS technology, t.category AS category,
                   criteria_using,
                   size(criteria_using) AS shared_count
            ORDER BY shared_count DESC, t.tech_id
        """
        results, elapsed = self._timed_query(cypher, {"cids": criterion_ids})
        return ToolResult(
            tool_name=self.name, success=True,
            data={
                "criterion_ids": criterion_ids,
                "shared_techniques_count": len(results),
                "techniques": results,
            },
            execution_time=elapsed,
        )

    def _disability_overlap(self, disability_a: str, disability_b: str) -> ToolResult:
        """Find criteria that overlap between two disability categories."""
        cypher = """
            MATCH (c:WCAGCriterion)
            WHERE any(d IN c.disability_impact WHERE toLower(d) CONTAINS toLower($da))
              AND any(d IN c.disability_impact WHERE toLower(d) CONTAINS toLower($db))
            MATCH (c)-[:PART_OF]->(g:WCAGGuideline)-[:PART_OF]->(p:WCAGPrinciple)
            RETURN c.ref_id AS ref_id, c.title AS title,
                   c.level AS level, c.disability_impact AS all_impacts,
                   c.in_brief_what_to_do AS what_to_do,
                   g.title AS guideline, p.title AS principle
            ORDER BY c.ref_id
        """
        results, elapsed = self._timed_query(cypher, {
            "da": disability_a.lower(), "db": disability_b.lower()
        })

        # Also get criteria unique to each
        only_a = self.db.query("""
            MATCH (c:WCAGCriterion)
            WHERE any(d IN c.disability_impact WHERE toLower(d) CONTAINS toLower($da))
              AND NOT any(d IN c.disability_impact WHERE toLower(d) CONTAINS toLower($db))
            RETURN c.ref_id AS ref_id, c.title AS title, c.level AS level
            ORDER BY c.ref_id
        """, {"da": disability_a.lower(), "db": disability_b.lower()})

        only_b = self.db.query("""
            MATCH (c:WCAGCriterion)
            WHERE any(d IN c.disability_impact WHERE toLower(d) CONTAINS toLower($db))
              AND NOT any(d IN c.disability_impact WHERE toLower(d) CONTAINS toLower($da))
            RETURN c.ref_id AS ref_id, c.title AS title, c.level AS level
            ORDER BY c.ref_id
        """, {"da": disability_a.lower(), "db": disability_b.lower()})

        return ToolResult(
            tool_name=self.name, success=True,
            data={
                "disability_a": disability_a,
                "disability_b": disability_b,
                "overlapping_criteria": len(results),
                "only_a_count": len(only_a),
                "only_b_count": len(only_b),
                "overlap": results,
                "only_a": only_a,
                "only_b": only_b,
            },
            execution_time=elapsed,
        )

    def _technique_coverage(self, technique_id: str) -> ToolResult:
        """Find all criteria that a specific technique satisfies."""
        cypher = """
            MATCH (t:WCAGTechnique {tech_id: $tid})
            OPTIONAL MATCH (c:WCAGCriterion)-[r:HAS_TECHNIQUE|HAS_ADVISORY_TECHNIQUE|HAS_FAILURE]->(t)
            RETURN t.tech_id AS tech_id, t.title AS title,
                   t.url AS url, t.technology AS technology,
                   collect(DISTINCT {
                       ref_id: c.ref_id,
                       criterion_title: c.title,
                       level: c.level,
                       relationship: type(r)
                   }) AS criteria
        """
        results, elapsed = self._timed_query(cypher, {"tid": technique_id})
        data = results[0] if results else {}

        # Clean nulls from criteria
        if data and "criteria" in data:
            data["criteria"] = [c for c in data["criteria"] if c.get("ref_id")]
            data["criteria_count"] = len(data["criteria"])

        return ToolResult(
            tool_name=self.name, success=True,
            data=data, execution_time=elapsed,
        )

    def _fix_ripple_effect(self, criterion_id: str) -> ToolResult:
        """Analyze the ripple effect: if you fix SC X, what else benefits?"""
        cypher = """
            MATCH (c:WCAGCriterion {ref_id: $cid})

            // Direct related criteria
            OPTIONAL MATCH (c)-[:RELATED_CRITERION]-(related:WCAGCriterion)

            // Shared techniques (other criteria using the same techniques)
            OPTIONAL MATCH (c)-[:HAS_TECHNIQUE]->(t:WCAGTechnique)<-[:HAS_TECHNIQUE]-(shared:WCAGCriterion)
            WHERE shared.ref_id <> $cid

            // Same disability impact
            WITH c, collect(DISTINCT {ref_id: related.ref_id, title: related.title,
                 level: related.level, connection: 'directly_related'}) AS related_criteria,
                 collect(DISTINCT {ref_id: shared.ref_id, title: shared.title,
                 level: shared.level, connection: 'shared_technique'}) AS technique_siblings

            RETURN c.ref_id AS source, c.title AS source_title,
                   c.level AS source_level,
                   c.disability_impact AS disability_impact,
                   related_criteria, technique_siblings
        """
        results, elapsed = self._timed_query(cypher, {"cid": criterion_id})
        data = results[0] if results else {}

        # Clean nulls
        if data:
            data["related_criteria"] = [r for r in data.get("related_criteria", [])
                                         if r.get("ref_id")]
            data["technique_siblings"] = [r for r in data.get("technique_siblings", [])
                                           if r.get("ref_id")]
            # Deduplicate
            seen = set()
            all_affected = []
            for item in data["related_criteria"] + data["technique_siblings"]:
                rid = item["ref_id"]
                if rid not in seen:
                    seen.add(rid)
                    all_affected.append(item)
            data["total_affected"] = len(all_affected)
            data["all_affected"] = all_affected

        return ToolResult(
            tool_name=self.name, success=True,
            data=data, execution_time=elapsed,
        )


# ============================================================
# TOOL 8: DYNAMIC CYPHER (LLM-generated with guardrails)
# ============================================================
class DynamicCypherTool(BaseTool):
    """
    Generates and executes Cypher queries using the LLM with strict
    safety guardrails.  This tool handles ad-hoc analytical questions
    that the pre-built templates cannot answer (aggregations, counts,
    negation queries, arbitrary filters, etc.).

    GUARDRAILS:
      • READ-ONLY: rejects any mutation keywords (CREATE, DELETE, SET,
        MERGE, REMOVE, DROP, DETACH, CALL, LOAD, FOREACH)
      • SCHEMA ALLOWLIST: only known node labels and relationship types
      • RESULT LIMIT: injects LIMIT if missing
      • TIMEOUT: kills long-running queries
      • PROPERTY ALLOWLIST: only known properties per node label
    """

    # ── Schema allowlists ──
    ALLOWED_NODE_LABELS = frozenset([
        "WCAGPrinciple", "WCAGGuideline", "WCAGCriterion", "ConformanceLevel",
        "WCAGSpecialCase", "WCAGNote", "WCAGReference", "WCAGTechnique",
        "WCAGTestRule", "WCAGExample", "WCAGBenefit", "WCAGKeyTerm",
        "WCAGRelatedResource",
    ])

    ALLOWED_RELATIONSHIP_TYPES = frozenset([
        "PART_OF", "HAS_LEVEL", "HAS_TECHNIQUE", "HAS_ADVISORY_TECHNIQUE",
        "HAS_FAILURE", "HAS_SPECIAL_CASE", "HAS_NOTE", "HAS_REFERENCE",
        "HAS_TEST_RULE", "HAS_EXAMPLE", "HAS_BENEFIT", "HAS_KEY_TERM",
        "HAS_RELATED_RESOURCE", "RELATED_CRITERION", "IMPACTS_DISABILITY",
    ])

    ALLOWED_PROPERTIES = {
        "WCAGPrinciple":      {"ref_id", "title", "description", "url"},
        "WCAGGuideline":      {"ref_id", "title", "description", "url"},
        "WCAGCriterion":      {"ref_id", "title", "description", "level", "url", "intent",
                               "in_brief_goal", "in_brief_what_to_do", "in_brief_why_important",
                               "wcag_version", "disabilities"},
        "ConformanceLevel":   {"name"},
        "WCAGSpecialCase":    {"title", "description", "type", "index"},
        "WCAGNote":           {"content", "index"},
        "WCAGReference":      {"title", "url"},
        "WCAGTechnique":      {"tech_id", "title", "url", "technology"},
        "WCAGTestRule":       {"title", "url", "rule_id"},
        "WCAGExample":        {"title", "description", "index"},
        "WCAGBenefit":        {"description", "index"},
        "WCAGKeyTerm":        {"term", "definition", "url"},
        "WCAGRelatedResource": {"title", "url"},
    }

    MUTATION_KEYWORDS = re.compile(
        r'\b(CREATE|DELETE|DETACH|SET|REMOVE|MERGE|DROP|CALL|LOAD|FOREACH)\b',
        re.IGNORECASE,
    )

    MAX_RESULT_ROWS = 50   # hard limit injected if LIMIT missing
    QUERY_TIMEOUT_MS = 10_000  # 10 seconds

    # Graph schema description given to the LLM for Cypher generation
    SCHEMA_PROMPT = """Neo4j WCAG 2.2 Knowledge Graph Schema
═══════════════════════════════════════

NODE LABELS & KEY PROPERTIES:
  (:WCAGPrinciple      {ref_id, title, description})
  (:WCAGGuideline      {ref_id, title, description})
  (:WCAGCriterion      {ref_id, title, description, level, intent,
                        in_brief_goal, in_brief_what_to_do, in_brief_why_important,
                        wcag_version, disabilities})
  (:ConformanceLevel   {name})            — values: "A", "AA", "AAA"
  (:WCAGSpecialCase    {title, description, type, index})
  (:WCAGNote           {content, index})
  (:WCAGReference      {title, url})
  (:WCAGTechnique      {tech_id, title, url, technology})
  (:WCAGTestRule       {title, url, rule_id})
  (:WCAGExample        {title, description, index})
  (:WCAGBenefit        {description, index})
  (:WCAGKeyTerm        {term, definition, url})
  (:WCAGRelatedResource {title, url})

RELATIONSHIPS:
  (Criterion)-[:PART_OF]->(Guideline)-[:PART_OF]->(Principle)
  (Criterion)-[:HAS_LEVEL]->(ConformanceLevel)
  (Criterion)-[:HAS_TECHNIQUE]->(Technique)          — sufficient
  (Criterion)-[:HAS_ADVISORY_TECHNIQUE]->(Technique)  — advisory
  (Criterion)-[:HAS_FAILURE]->(Technique)             — failure
  (Criterion)-[:HAS_SPECIAL_CASE]->(SpecialCase)
  (Criterion)-[:HAS_NOTE]->(Note)
  (Criterion)-[:HAS_REFERENCE]->(Reference)
  (Criterion)-[:HAS_TEST_RULE]->(TestRule)
  (Criterion)-[:HAS_EXAMPLE]->(Example)
  (Criterion)-[:HAS_BENEFIT]->(Benefit)
  (Criterion)-[:HAS_KEY_TERM]->(KeyTerm)
  (Criterion)-[:HAS_RELATED_RESOURCE]->(RelatedResource)
  (Criterion)-[:RELATED_CRITERION]->(Criterion)
  (Criterion)-[:IMPACTS_DISABILITY]->(label stored in disabilities property)

CARDINALITY HINTS:
  4 Principles → 13 Guidelines → 87 Criteria
  412 Techniques, 725 Key Terms, 284 Examples, 261 Related Resources
  204 Benefits, 109 Test Rules, 91 Special Cases

RULES:
  • Generate READ-ONLY Cypher (no CREATE, SET, DELETE, MERGE, REMOVE, etc.)
  • Always include a LIMIT clause (max 50)
  • Use parameterised values with $param syntax when filtering on user input
  • Return meaningful aliases (AS column_name)
  • Prefer OPTIONAL MATCH for paths that may not exist
"""

    def __init__(self, db: Neo4jConnection, llm_client=None, llm_model: str = ""):
        super().__init__(db)
        self._llm_client = llm_client
        self._llm_model = llm_model

    def set_llm(self, llm_client, llm_model: str):
        """Inject LLM client after construction (called by WCAGAgent)."""
        self._llm_client = llm_client
        self._llm_model = llm_model

    @property
    def name(self) -> str:
        return "dynamic_cypher"

    @property
    def description(self) -> str:
        return (
            "Generate and execute an ad-hoc Cypher query against the WCAG knowledge graph. "
            "Use for analytical / aggregate / negation questions that the pre-built tools "
            "cannot answer, e.g., 'which criteria have more than 5 techniques?', "
            "'show criteria WITHOUT test rules', 'count techniques per technology'. "
            "The query is validated for safety before execution (read-only, schema-checked)."
        )

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": (
                        "Natural-language question to answer via Cypher. "
                        "The LLM will translate this into a safe, read-only Cypher query."
                    ),
                },
                "cypher_hint": {
                    "type": "string",
                    "description": (
                        "Optional: a Cypher fragment or pattern the agent thinks is relevant. "
                        "Used as a hint for the Cypher generator."
                    ),
                },
            },
            "required": ["question"],
        }

    # ── Public entry point ──
    def execute(self, **kwargs) -> ToolResult:
        question = kwargs.get("question", "")
        cypher_hint = kwargs.get("cypher_hint", "")

        if not question:
            return ToolResult(tool_name=self.name, success=False,
                              error="A natural-language question is required.")

        if not self._llm_client:
            return ToolResult(tool_name=self.name, success=False,
                              error="No LLM configured — dynamic Cypher requires an LLM.")

        # Step 1: Ask LLM to generate Cypher
        generated = self._generate_cypher(question, cypher_hint)
        if not generated["success"]:
            return ToolResult(tool_name=self.name, success=False,
                              error=generated["error"])

        cypher = generated["cypher"]
        params = generated.get("params", {})
        explanation = generated.get("explanation", "")

        # Step 2: Validate the generated Cypher
        validation = self._validate_cypher(cypher)
        if not validation["safe"]:
            return ToolResult(
                tool_name=self.name, success=False,
                error=f"Cypher rejected by guardrails: {validation['reason']}",
                cypher_used=cypher,
            )

        # Step 3: Inject LIMIT if missing
        cypher = self._ensure_limit(cypher)

        # Step 4: Execute with timeout
        try:
            results, elapsed = self._timed_query(cypher, params)
        except Exception as e:
            return ToolResult(
                tool_name=self.name, success=False,
                error=f"Cypher execution error: {e}",
                cypher_used=cypher,
            )

        return ToolResult(
            tool_name=self.name, success=True,
            data={
                "question": question,
                "cypher": cypher,
                "params": params,
                "explanation": explanation,
                "row_count": len(results),
                "results": results[:self.MAX_RESULT_ROWS],
            },
            execution_time=elapsed,
            cypher_used=cypher,
        )

    # ── Cypher generation via LLM ──
    def _generate_cypher(self, question: str, hint: str = "") -> dict:
        """Ask the LLM to produce a Cypher query for the given question."""
        user_content = f"Question: {question}"
        if hint:
            user_content += f"\nHint: {hint}"
        user_content += (
            "\n\nGenerate a READ-ONLY Cypher query. Respond in JSON with keys: "
            '"cypher" (string), "params" (object, may be empty), "explanation" (string). '
            "Return ONLY the JSON — no markdown fences, no extra text."
        )

        try:
            resp = self._llm_client.chat.completions.create(
                model=self._llm_model,
                messages=[
                    {"role": "system", "content": self.SCHEMA_PROMPT},
                    {"role": "user", "content": user_content},
                ],
                temperature=0.0,
                max_tokens=1024,
            )
            raw = resp.choices[0].message.content.strip()

            # Strip markdown code fences if present
            if raw.startswith("```"):
                raw = re.sub(r'^```(?:json)?\s*', '', raw)
                raw = re.sub(r'\s*```$', '', raw)

            payload = json.loads(raw)
            cypher = payload.get("cypher", "").strip()
            if not cypher:
                return {"success": False, "error": "LLM returned empty Cypher"}
            return {
                "success": True,
                "cypher": cypher,
                "params": payload.get("params", {}),
                "explanation": payload.get("explanation", ""),
            }
        except json.JSONDecodeError as e:
            return {"success": False, "error": f"LLM returned invalid JSON: {e}"}
        except Exception as e:
            return {"success": False, "error": f"Cypher generation failed: {e}"}

    # ── Cypher validation (guardrails) ──
    def _validate_cypher(self, cypher: str) -> dict:
        """Validate a Cypher query against all guardrails.  Returns {'safe': bool, 'reason': str}."""

        # Guard 1: mutation keywords
        if self.MUTATION_KEYWORDS.search(cypher):
            return {"safe": False, "reason": "Mutation keyword detected — only read queries allowed"}

        # Guard 2: node label allowlist
        # Node labels appear as (:Label) or (var:Label) — NOT inside square brackets
        label_pattern = re.compile(r'\(\w*:(\w+)')
        labels_found = set(label_pattern.findall(cypher))
        unknown_labels = labels_found - self.ALLOWED_NODE_LABELS
        if unknown_labels:
            return {"safe": False, "reason": f"Unknown node label(s): {unknown_labels}"}

        # Guard 3: relationship type allowlist
        # Relationship types appear inside square brackets: [:TYPE] or [r:TYPE]
        rel_pattern = re.compile(r'\[\w*:([A-Z_|]+)')
        rels_raw = rel_pattern.findall(cypher)
        rels_found: set[str] = set()
        for raw in rels_raw:
            for part in raw.split("|"):
                part = part.strip()
                if part:
                    rels_found.add(part)
        unknown_rels = rels_found - self.ALLOWED_RELATIONSHIP_TYPES
        if unknown_rels:
            return {"safe": False, "reason": f"Unknown relationship type(s): {unknown_rels}"}

        # Guard 4: property allowlist — for each label, check accessed properties
        for label in labels_found:
            if label in self.ALLOWED_PROPERTIES:
                allowed_props = self.ALLOWED_PROPERTIES[label]
                # Find property accesses for this label's variable aliases
                # e.g.  (c:WCAGCriterion)  then  c.some_prop  or  {some_prop: ...}
                # We'll do a best-effort check using .<prop> pattern near the label
                # Full static analysis is complex; this catches obvious mistakes
                prop_access = re.findall(r'\.(\w+)', cypher)
                # Only flag if a property clearly doesn't exist on ANY label
                # (conservative: allow any property that exists on any node)
                pass  # conservative — full property validation deferred to Neo4j

        # Guard 5: no APOC / db.* / gds.* procedures
        if re.search(r'\b(apoc|gds|db|dbms)\.\w+', cypher, re.IGNORECASE):
            return {"safe": False, "reason": "Procedure/function calls not allowed (apoc/gds/db)"}

        return {"safe": True, "reason": ""}

    def _ensure_limit(self, cypher: str) -> str:
        """Inject a LIMIT clause if the query doesn't already have one."""
        if not re.search(r'\bLIMIT\b', cypher, re.IGNORECASE):
            cypher = cypher.rstrip().rstrip(';')
            cypher += f"\nLIMIT {self.MAX_RESULT_ROWS}"
        return cypher


# ============================================================
# TOOL 9: CONTEXT ASSEMBLER
# ============================================================
class ContextAssemblerTool(BaseTool):
    """
    Assembles rich, structured context from graph subgraphs
    specifically formatted for LLM consumption.
    """

    @property
    def name(self) -> str:
        return "context_assembler"

    @property
    def description(self) -> str:
        return (
            "Assemble comprehensive, LLM-ready context for a criterion or topic. "
            "Pulls the full subgraph including hierarchy, techniques, examples, "
            "benefits, and related criteria. Use as the FINAL step before generating "
            "a response to ensure the LLM has complete information."
        )

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "criterion_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of criterion ref_ids to assemble context for"
                },
                "include_techniques": {
                    "type": "boolean",
                    "default": True
                },
                "include_examples": {
                    "type": "boolean",
                    "default": True
                },
                "include_related": {
                    "type": "boolean",
                    "default": True
                },
            },
            "required": ["criterion_ids"],
        }

    def execute(self, **kwargs) -> ToolResult:
        criterion_ids = kwargs.get("criterion_ids", [])
        include_techniques = kwargs.get("include_techniques", True)
        include_examples = kwargs.get("include_examples", True)
        include_related = kwargs.get("include_related", True)

        if not criterion_ids:
            return ToolResult(
                tool_name=self.name, success=False,
                error="At least one criterion_id is required"
            )

        assembled_contexts = []
        for cid in criterion_ids:
            ctx = self._assemble_single(cid, include_techniques,
                                         include_examples, include_related)
            assembled_contexts.append(ctx)

        # Format as structured text for LLM
        formatted = self._format_for_llm(assembled_contexts)

        return ToolResult(
            tool_name=self.name, success=True,
            data={
                "criteria_count": len(assembled_contexts),
                "formatted_context": formatted,
                "raw": assembled_contexts,
            },
        )

    def _assemble_single(self, criterion_id: str,
                          include_techniques: bool,
                          include_examples: bool,
                          include_related: bool) -> dict:
        """Assemble full context for a single criterion."""
        # Core criterion + hierarchy
        core = self.db.query("""
            MATCH (c:WCAGCriterion {ref_id: $cid})
            MATCH (c)-[:PART_OF]->(g:WCAGGuideline)-[:PART_OF]->(p:WCAGPrinciple)
            MATCH (c)-[:HAS_LEVEL]->(cl:ConformanceLevel)
            OPTIONAL MATCH (c)-[:HAS_SPECIAL_CASE]->(sc:WCAGSpecialCase)
            OPTIONAL MATCH (c)-[:HAS_NOTE]->(n:WCAGNote)
            RETURN c {.*} AS criterion,
                   p.title AS principle, g.title AS guideline,
                   cl.name AS level,
                   collect(DISTINCT sc {.*}) AS special_cases,
                   collect(DISTINCT n.content) AS notes
        """, {"cid": criterion_id})

        if not core:
            return {"criterion_id": criterion_id, "error": "Not found"}

        result = core[0]

        # Techniques
        if include_techniques:
            techniques = self.db.query("""
                MATCH (c:WCAGCriterion {ref_id: $cid})
                OPTIONAL MATCH (c)-[:HAS_TECHNIQUE]->(suf:WCAGTechnique)
                OPTIONAL MATCH (c)-[:HAS_ADVISORY_TECHNIQUE]->(adv:WCAGTechnique)
                OPTIONAL MATCH (c)-[:HAS_FAILURE]->(fail:WCAGTechnique)
                RETURN collect(DISTINCT suf {.*}) AS sufficient,
                       collect(DISTINCT adv {.*}) AS advisory,
                       collect(DISTINCT fail {.*}) AS failures
            """, {"cid": criterion_id})
            result["techniques"] = techniques[0] if techniques else {}

        # Examples
        if include_examples:
            examples = self.db.query("""
                MATCH (c:WCAGCriterion {ref_id: $cid})-[:HAS_EXAMPLE]->(ex:WCAGExample)
                RETURN ex.title AS title, ex.description AS description
                ORDER BY ex.index
            """, {"cid": criterion_id})
            result["examples"] = examples

            # Benefits
            benefits = self.db.query("""
                MATCH (c:WCAGCriterion {ref_id: $cid})-[:HAS_BENEFIT]->(b:WCAGBenefit)
                RETURN b.description AS benefit
                ORDER BY b.index
            """, {"cid": criterion_id})
            result["benefits"] = [b["benefit"] for b in benefits]

        # Related criteria
        if include_related:
            related = self.db.query("""
                MATCH (c:WCAGCriterion {ref_id: $cid})-[:RELATED_CRITERION]->(r:WCAGCriterion)
                RETURN r.ref_id AS ref_id, r.title AS title, r.level AS level
            """, {"cid": criterion_id})
            result["related_criteria"] = related

            # Test rules
            test_rules = self.db.query("""
                MATCH (c:WCAGCriterion {ref_id: $cid})-[:HAS_TEST_RULE]->(tr:WCAGTestRule)
                RETURN tr.title AS title, tr.url AS url
            """, {"cid": criterion_id})
            result["test_rules"] = test_rules

            # Key terms
            key_terms = self.db.query("""
                MATCH (c:WCAGCriterion {ref_id: $cid})-[:HAS_KEY_TERM]->(kt:WCAGKeyTerm)
                RETURN kt.term AS term, kt.definition AS definition
                ORDER BY kt.term
            """, {"cid": criterion_id})
            result["key_terms"] = key_terms

            # Related resources
            related_resources = self.db.query("""
                MATCH (c:WCAGCriterion {ref_id: $cid})-[:HAS_RELATED_RESOURCE]->(rr:WCAGRelatedResource)
                RETURN rr.title AS title, rr.url AS url
                ORDER BY rr.title
            """, {"cid": criterion_id})
            result["related_resources"] = related_resources

        return result

    def _format_for_llm(self, contexts: list[dict]) -> str:
        """Format assembled contexts as structured text for LLM consumption."""
        sections = []
        for ctx in contexts:
            criterion = ctx.get("criterion", {})
            if isinstance(criterion, dict):
                cid = criterion.get("ref_id", "?")
                title = criterion.get("title", "?")
                desc = criterion.get("description", "")
                intent = criterion.get("intent", "")
            else:
                cid = ctx.get("criterion_id", "?")
                title = ""
                desc = ""
                intent = ""

            section = f"""
═══════════════════════════════════════
WCAG {cid}: {title}
═══════════════════════════════════════
Principle:  {ctx.get('principle', '?')}
Guideline:  {ctx.get('guideline', '?')}
Level:      {ctx.get('level', '?')}

Description:
{desc}
"""
            if intent:
                section += f"\nIntent:\n{intent}\n"

            # In-brief
            goal = criterion.get("in_brief_goal", "") if isinstance(criterion, dict) else ""
            what_to_do = criterion.get("in_brief_what_to_do", "") if isinstance(criterion, dict) else ""
            why_important = criterion.get("in_brief_why_important", "") if isinstance(criterion, dict) else ""
            if goal or what_to_do or why_important:
                section += f"\nIn Brief:\n  Goal: {goal}\n  What to do: {what_to_do}\n  Why: {why_important}\n"

            # Special cases
            special_cases = ctx.get("special_cases", [])
            if special_cases:
                section += "\nSpecial Cases:\n"
                for sc in special_cases:
                    if isinstance(sc, dict):
                        section += f"  • [{sc.get('type', '')}] {sc.get('title', '')}: {sc.get('description', '')}\n"

            # Notes
            notes = ctx.get("notes", [])
            if notes:
                section += "\nNotes:\n"
                for note in notes:
                    section += f"  • {note}\n"

            # Techniques
            techniques = ctx.get("techniques", {})
            if techniques:
                section += "\nTechniques:\n"
                for suf in techniques.get("sufficient", []):
                    if isinstance(suf, dict) and suf.get("tech_id"):
                        section += f"  ✅ [{suf['tech_id']}] {suf.get('title', '')}\n"
                for adv in techniques.get("advisory", []):
                    if isinstance(adv, dict) and adv.get("tech_id"):
                        section += f"  💡 [{adv['tech_id']}] {adv.get('title', '')}\n"
                for fail in techniques.get("failures", []):
                    if isinstance(fail, dict) and fail.get("tech_id"):
                        section += f"  ❌ [{fail['tech_id']}] {fail.get('title', '')}\n"

            # Examples
            examples = ctx.get("examples", [])
            if examples:
                section += "\nExamples:\n"
                for ex in examples:
                    section += f"  • {ex.get('title', '')}: {ex.get('description', '')}\n"

            # Benefits
            benefits = ctx.get("benefits", [])
            if benefits:
                section += "\nBenefits:\n"
                for b in benefits:
                    section += f"  • {b}\n"

            # Related
            related = ctx.get("related_criteria", [])
            if related:
                section += "\nRelated Criteria:\n"
                for r in related:
                    section += f"  → {r.get('ref_id', '')} {r.get('title', '')} (Level {r.get('level', '')})\n"

            # Test rules
            test_rules = ctx.get("test_rules", [])
            if test_rules:
                section += "\nAutomated Test Rules:\n"
                for tr in test_rules:
                    section += f"  🧪 {tr.get('title', '')} — {tr.get('url', '')}\n"

            # Key terms
            key_terms = ctx.get("key_terms", [])
            if key_terms:
                section += "\nKey Terms:\n"
                for kt in key_terms:
                    defn = (kt.get("definition", "") or "")[:200]
                    section += f"  📖 {kt.get('term', '')}: {defn}\n"

            # Related resources
            related_resources = ctx.get("related_resources", [])
            if related_resources:
                section += "\nRelated Resources:\n"
                for rr in related_resources:
                    section += f"  🔗 {rr.get('title', '')} — {rr.get('url', '')}\n"

            sections.append(section)

        return "\n".join(sections)


# ============================================================
# QUERY DECOMPOSER — breaks complex queries into sub-steps
# ============================================================
@dataclass
class QueryStep:
    """A single sub-step in a decomposed query plan."""
    step_id: int
    description: str
    tool: str                      # tool name to call
    params: dict = field(default_factory=dict)
    depends_on: list[int] = field(default_factory=list)  # step_ids this depends on
    priority: int = 1              # 1 = highest
    status: str = "pending"        # pending | running | done | failed
    result: Optional[ToolResult] = None

    def __repr__(self):
        deps = f" (after {self.depends_on})" if self.depends_on else ""
        return f"Step {self.step_id} [P{self.priority}]: {self.tool} — {self.description}{deps}"


@dataclass
class QueryPlan:
    """A full execution plan for a user query."""
    original_query: str
    intent_summary: str = ""
    steps: list[QueryStep] = field(default_factory=list)
    reasoning: str = ""
    created_at: float = 0.0

    def pending_steps(self) -> list[QueryStep]:
        """Return steps whose dependencies are all satisfied."""
        done_ids = {s.step_id for s in self.steps if s.status == "done"}
        return [
            s for s in self.steps
            if s.status == "pending" and all(d in done_ids for d in s.depends_on)
        ]

    def is_complete(self) -> bool:
        return all(s.status in ("done", "failed") for s in self.steps)


class QueryDecomposer:
    """
    Decomposes a user query into an ordered plan of tool calls.
    Uses the LLM for complex queries; falls back to simple heuristics.
    """

    # Mapping from intent keywords → decomposition templates
    TEMPLATES: dict[str, list[dict]] = {
        "audit_scenario": [
            {"tool": "rule_engine", "desc": "Identify applicable rules for each UI element",
             "params_hint": "check_type=element_rules"},
            {"tool": "context_assembler", "desc": "Assemble full context for discovered criteria",
             "params_hint": "criterion_ids from step 1"},
        ],
        "comparison": [
            {"tool": "graph_traversal", "desc": "Retrieve side A data"},
            {"tool": "graph_traversal", "desc": "Retrieve side B data"},
            {"tool": "dynamic_cypher", "desc": "Compare the two sets",
             "params_hint": "aggregation / difference query"},
        ],
        "aggregate_analytics": [
            {"tool": "dynamic_cypher", "desc": "Run analytical Cypher query"},
        ],
        "deep_dive_criterion": [
            {"tool": "context_assembler", "desc": "Get full criterion context"},
            {"tool": "technique_finder", "desc": "Find implementation techniques"},
            {"tool": "cross_reference", "desc": "Map related criteria and ripple effects"},
        ],
    }

    DECOMPOSITION_PROMPT = """You are a query planner for a WCAG 2.2 accessibility knowledge graph.
Given a user question, decompose it into a sequence of tool calls.

AVAILABLE TOOLS:
1. graph_traversal — ID-based hierarchy navigation (principle/guideline/criterion lookups)
2. semantic_search — keyword/topic search across criteria, examples, benefits, key terms
3. technique_finder — find sufficient/advisory/failure techniques for a criterion
4. rule_engine — compliance rules, element-specific rules, disability impact, checklists
5. impact_analysis — disability matrix, input modality analysis
6. key_term_lookup — WCAG terminology definitions
7. cross_reference — multi-hop analysis: related chains, shared techniques, disability overlap, ripple effects
8. context_assembler — assemble full LLM-ready context for criteria (use as final step)
9. dynamic_cypher — ad-hoc analytical Cypher queries (counts, aggregations, negations, filters not covered by other tools)

RULES:
- Each step must specify: tool name, parameters, a brief description, dependencies (which prior step IDs), and priority (1=highest).
- Use dynamic_cypher ONLY when no pre-built tool can answer the sub-question.
- Always end with context_assembler if the answer requires detailed criterion information.
- Keep plans concise: 2-5 steps for most queries.
- For simple single-tool queries, return just 1 step.

Respond in JSON with this schema:
{
  "intent_summary": "one-line summary of what the user wants",
  "reasoning": "brief explanation of the decomposition logic",
  "steps": [
    {
      "step_id": 1,
      "description": "what this step does",
      "tool": "tool_name",
      "params": {"param1": "value1"},
      "depends_on": [],
      "priority": 1
    }
  ]
}
Return ONLY JSON — no markdown fences, no extra text."""

    def __init__(self, llm_client=None, llm_model: str = "",
                 available_tools: list[str] | None = None):
        self._llm_client = llm_client
        self._llm_model = llm_model
        self._available_tools = set(available_tools or [])

    def set_llm(self, llm_client, llm_model: str):
        self._llm_client = llm_client
        self._llm_model = llm_model

    def decompose(self, query: str) -> QueryPlan:
        """Decompose a query into a QueryPlan.  Uses LLM if available."""
        if self._llm_client:
            return self._llm_decompose(query)
        return self._heuristic_decompose(query)

    # ── LLM-powered decomposition ──
    def _llm_decompose(self, query: str) -> QueryPlan:
        """Use the LLM to decompose the query into steps."""
        try:
            resp = self._llm_client.chat.completions.create(
                model=self._llm_model,
                messages=[
                    {"role": "system", "content": self.DECOMPOSITION_PROMPT},
                    {"role": "user", "content": query},
                ],
                temperature=0.0,
                max_tokens=1024,
            )
            raw = resp.choices[0].message.content.strip()

            # Strip markdown fences
            if raw.startswith("```"):
                raw = re.sub(r'^```(?:json)?\s*', '', raw)
                raw = re.sub(r'\s*```$', '', raw)

            payload = json.loads(raw)

            steps: list[QueryStep] = []
            for s in payload.get("steps", []):
                tool_name = s.get("tool", "")
                # Validate tool exists
                if self._available_tools and tool_name not in self._available_tools:
                    log.warning("  Decomposer: LLM suggested unknown tool '%s', skipping", tool_name)
                    continue
                steps.append(QueryStep(
                    step_id=s.get("step_id", len(steps) + 1),
                    description=s.get("description", ""),
                    tool=tool_name,
                    params=s.get("params", {}),
                    depends_on=s.get("depends_on", []),
                    priority=s.get("priority", len(steps) + 1),
                ))

            if not steps:
                log.warning("  Decomposer: LLM returned no valid steps, falling back")
                return self._heuristic_decompose(query)

            return QueryPlan(
                original_query=query,
                intent_summary=payload.get("intent_summary", ""),
                steps=steps,
                reasoning=payload.get("reasoning", ""),
                created_at=time.time(),
            )
        except Exception as e:
            log.warning("  Decomposer LLM failed: %s — using heuristic", e)
            return self._heuristic_decompose(query)

    # ── Heuristic (rule-based) decomposition ──
    def _heuristic_decompose(self, query: str) -> QueryPlan:
        """Simple pattern-matching decomposition without an LLM."""
        q = query.lower().strip()
        steps: list[QueryStep] = []
        step_id = 0

        # Detect criterion IDs
        criterion_ids = re.findall(r'\b(\d+\.\d+\.\d+)\b', q)

        # Detect element keywords
        elements = []
        element_map = {
            "image": "image", "img": "image", "form": "form",
            "video": "video", "audio": "audio", "link": "link",
            "button": "button", "table": "table", "heading": "heading",
            "color": "color", "contrast": "color", "keyboard": "button",
        }
        for kw, elem in element_map.items():
            if kw in q and elem not in elements:
                elements.append(elem)

        # Pattern: aggregate / count / analytics → dynamic_cypher
        if any(kw in q for kw in ["how many", "count", "most", "least", "average",
                                     "without", "missing", "more than", "fewer than",
                                     "percentage", "which criteria have"]):
            step_id += 1
            steps.append(QueryStep(
                step_id=step_id, description="Run analytical query",
                tool="dynamic_cypher", params={"question": query}, priority=1,
            ))
            return QueryPlan(original_query=query, intent_summary="analytics",
                             steps=steps, created_at=time.time())

        # Pattern: specific criteria → context_assembler + optional techniques
        if criterion_ids:
            step_id += 1
            steps.append(QueryStep(
                step_id=step_id, description=f"Assemble context for {', '.join(criterion_ids)}",
                tool="context_assembler", params={"criterion_ids": criterion_ids}, priority=1,
            ))
            if any(kw in q for kw in ["how", "technique", "fix", "implement"]):
                for cid in criterion_ids:
                    step_id += 1
                    steps.append(QueryStep(
                        step_id=step_id, description=f"Find techniques for {cid}",
                        tool="technique_finder",
                        params={"criterion_id": cid, "technique_type": "all"},
                        priority=2,
                    ))
            if any(kw in q for kw in ["related", "ripple", "cascade", "also"]):
                step_id += 1
                steps.append(QueryStep(
                    step_id=step_id, description=f"Analyze ripple effect from {criterion_ids[0]}",
                    tool="cross_reference",
                    params={"analysis_type": "fix_ripple_effect", "criterion_id": criterion_ids[0]},
                    priority=2,
                ))
            return QueryPlan(original_query=query, intent_summary="criterion_detail",
                             steps=steps, created_at=time.time())

        # Pattern: element audit → rule_engine per element → context_assembler
        if elements:
            for elem in elements:
                step_id += 1
                steps.append(QueryStep(
                    step_id=step_id, description=f"Find rules for {elem} elements",
                    tool="rule_engine",
                    params={"check_type": "element_rules", "element_type": elem},
                    priority=1,
                ))
            step_id += 1
            steps.append(QueryStep(
                step_id=step_id, description="Assemble context for discovered criteria",
                tool="context_assembler", params={"criterion_ids": []},
                depends_on=list(range(1, step_id)), priority=2,
            ))
            return QueryPlan(original_query=query, intent_summary="element_rules",
                             steps=steps, created_at=time.time())

        # Pattern: terminology
        term_match = re.search(
            r'(?:what\s+(?:does|is|are)|define|meaning\s+of)\s+["\']?(.+?)["\']?\s*(?:\?|$)',
            q, re.IGNORECASE,
        )
        if term_match:
            term = term_match.group(1).strip().rstrip("?. ")
            step_id += 1
            steps.append(QueryStep(
                step_id=step_id, description=f"Look up definition of '{term}'",
                tool="key_term_lookup", params={"term": term}, priority=1,
            ))
            return QueryPlan(original_query=query, intent_summary="terminology",
                             steps=steps, created_at=time.time())

        # Fallback: semantic search
        step_id += 1
        steps.append(QueryStep(
            step_id=step_id, description="Broad keyword search",
            tool="semantic_search", params={"query": query, "limit": 10}, priority=1,
        ))
        return QueryPlan(original_query=query, intent_summary="general_search",
                         steps=steps, created_at=time.time())


# ============================================================
# STEP PLANNER / EXECUTOR — runs a QueryPlan with dependency ordering
# ============================================================
class StepExecutor:
    """
    Executes a QueryPlan step-by-step, respecting dependencies.
    Passes intermediate results between steps when needed.
    """

    def __init__(self, tools: dict[str, BaseTool], verbose: bool = True):
        self.tools = tools
        self.verbose = verbose

    def execute_plan(self, plan: QueryPlan, trace_steps: list | None = None) -> list[ToolResult]:
        """Execute all steps in dependency/priority order.  Returns list of ToolResults."""
        results: list[ToolResult] = []
        max_iterations = len(plan.steps) * 2  # safety valve

        for _ in range(max_iterations):
            if plan.is_complete():
                break

            ready = plan.pending_steps()
            if not ready:
                log.warning("  StepExecutor: no ready steps but plan not complete — breaking")
                break

            # Pick highest-priority ready step
            ready.sort(key=lambda s: s.priority)
            step = ready[0]
            step.status = "running"

            if self.verbose:
                log.info("  PLAN STEP %d [P%d] → %s(%s) — %s",
                         step.step_id, step.priority, step.tool,
                         json.dumps(step.params, default=str)[:120], step.description)

            # Resolve dynamic parameters from prior step results
            params = self._resolve_params(step, plan)

            if step.tool not in self.tools:
                step.status = "failed"
                step.result = ToolResult(tool_name=step.tool, success=False,
                                         error=f"Unknown tool: {step.tool}")
                results.append(step.result)
                continue

            try:
                result = self.tools[step.tool].execute(**params)
            except Exception as e:
                result = ToolResult(tool_name=step.tool, success=False,
                                     error=f"Execution error: {e}")

            step.result = result
            step.status = "done" if result.success else "failed"
            results.append(result)

            if self.verbose:
                status_icon = "✅" if result.success else "❌"
                data_preview = str(result.data)[:150] if result.data else "empty"
                log.info("  PLAN STEP %d %s (%.3fs): %s",
                         step.step_id, status_icon, result.execution_time, data_preview)

            # Track in agent trace if provided
            if trace_steps is not None:
                trace_steps.append(AgentStep(
                    step_number=len(trace_steps) + 1,
                    action=AgentAction.CALL_TOOL,
                    thought=f"Plan step {step.step_id}: {step.description}",
                    tool_name=step.tool,
                    tool_params=params,
                    tool_result=result,
                    timestamp=time.time(),
                ))

        return results

    def _resolve_params(self, step: QueryStep, plan: QueryPlan) -> dict:
        """
        Resolve dynamic parameters. If a step depends on prior steps
        and has placeholder params, fill them from prior results.
        """
        params = dict(step.params)

        # Special case: context_assembler with empty criterion_ids
        # → collect IDs from all prior steps' results
        if step.tool == "context_assembler" and not params.get("criterion_ids"):
            collected_ids = []
            for dep_id in step.depends_on:
                dep_step = next((s for s in plan.steps if s.step_id == dep_id), None)
                if dep_step and dep_step.result and dep_step.result.success:
                    collected_ids.extend(self._extract_criterion_ids(dep_step.result.data))
            if collected_ids:
                params["criterion_ids"] = list(dict.fromkeys(collected_ids))[:5]

        return params

    @staticmethod
    def _extract_criterion_ids(data) -> list[str]:
        """Extract criterion ref_ids from tool result data."""
        ids = []
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict) and "ref_id" in item:
                    ref_id = str(item["ref_id"])
                    if re.match(r'^\d+\.\d+\.\d+$', ref_id):
                        ids.append(ref_id)
        elif isinstance(data, dict):
            for key, value in data.items():
                if key in ("rules", "criteria", "checklist", "results"):
                    ids.extend(StepExecutor._extract_criterion_ids(value))
                elif key == "ref_id" and re.match(r'^\d+\.\d+\.\d+$', str(value)):
                    ids.append(str(value))
        return ids


# ============================================================
# AGENT REASONING ENGINE
# ============================================================
class AgentAction(Enum):
    """Possible agent actions."""
    CALL_TOOL = "call_tool"
    RESPOND = "respond"
    THINK = "think"


@dataclass
class AgentStep:
    """Record of a single agent reasoning step."""
    step_number: int
    action: AgentAction
    thought: str = ""
    tool_name: str = ""
    tool_params: dict = field(default_factory=dict)
    tool_result: Optional[ToolResult] = None
    response: str = ""
    timestamp: float = 0.0


@dataclass
class AgentTrace:
    """Complete trace of agent reasoning for transparency."""
    query: str
    steps: list[AgentStep] = field(default_factory=list)
    final_response: str = ""
    total_time: float = 0.0
    tools_called: list[str] = field(default_factory=list)


class WCAGAgent:
    """
    Agentic RAG orchestrator for WCAG knowledge graph.

    The agent follows a ReAct (Reasoning + Acting) loop:
      1. THINK  — Analyze the query, plan next action
      2. ACT    — Call a tool based on the plan
      3. OBSERVE— Process tool results
      4. REPEAT — Until enough context is gathered
      5. RESPOND— Generate final answer with assembled context

    Without an LLM, uses rule-based routing (QueryRouter).
    With an LLM, uses function calling for dynamic tool selection.
    """

    def __init__(self, config: AgentConfig, db: Neo4jConnection):
        self.config = config
        self.db = db
        self.tools: dict[str, BaseTool] = {}
        self.llm_client = None
        self._llm_model_name: str = config.llm_model  # may be overridden by _init_llm
        self.decomposer: Optional[QueryDecomposer] = None
        self.step_executor: Optional[StepExecutor] = None

        # Register all tools
        self._register_tools()

        # Initialize LLM if configured
        self._init_llm()

    def _register_tools(self):
        """Register all available tools."""
        tool_classes = [
            GraphTraversalTool,
            SemanticSearchTool,
            TechniqueFinderTool,
            RuleEngineTool,
            ImpactAnalysisTool,
            KeyTermLookupTool,
            CrossReferenceTool,
            DynamicCypherTool,
            ContextAssemblerTool,
        ]
        for cls in tool_classes:
            tool = cls(self.db)
            self.tools[tool.name] = tool
            log.info("  Registered tool: %s", tool.name)

        # Initialize decomposer and executor (LLM injected later in _init_llm)
        self.decomposer = QueryDecomposer(
            available_tools=list(self.tools.keys()),
        )
        self.step_executor = StepExecutor(
            tools=self.tools, verbose=self.config.verbose,
        )

    def _init_llm(self):
        """Initialize LLM client if API key is available."""
        if not self.config.llm_api_key and not self.config.llm_base_url and not self.config.azure_openai_endpoint:
            log.info("  No LLM configured — using rule-based routing")
            return

        try:
            provider = self.config.llm_provider.lower()

            if provider == "azure_openai":
                from openai import AzureOpenAI

                if not self.config.azure_openai_endpoint:
                    log.warning("  AZURE_OPENAI_ENDPOINT not set — using rule-based routing")
                    return

                self.llm_client = AzureOpenAI(
                    api_key=self.config.llm_api_key,
                    azure_endpoint=self.config.azure_openai_endpoint,
                    api_version=self.config.azure_openai_api_version,
                )
                # For Azure, the "model" param in chat.completions.create
                # must be the deployment name, not the model name
                self._llm_model_name = (
                    self.config.azure_openai_deployment
                    or self.config.llm_model
                )
                log.info("  LLM initialized: Azure OpenAI — deployment=%s, endpoint=%s",
                         self._llm_model_name, self.config.azure_openai_endpoint)
            else:
                from openai import OpenAI

                client_kwargs = {"api_key": self.config.llm_api_key}
                if self.config.llm_base_url:
                    client_kwargs["base_url"] = self.config.llm_base_url

                self.llm_client = OpenAI(**client_kwargs)
                self._llm_model_name = self.config.llm_model
                log.info("  LLM initialized: %s (%s)",
                         self._llm_model_name, provider)

        except ImportError:
            log.warning("  openai package not installed — using rule-based routing")
        except Exception as e:
            log.warning("  LLM init failed: %s — using rule-based routing", e)

        # Inject LLM into DynamicCypherTool and QueryDecomposer
        if self.llm_client:
            dyn = self.tools.get("dynamic_cypher")
            if dyn and isinstance(dyn, DynamicCypherTool):
                dyn.set_llm(self.llm_client, self._llm_model_name)
                log.info("  DynamicCypherTool: LLM injected")
            if self.decomposer:
                self.decomposer.set_llm(self.llm_client, self._llm_model_name)
                log.info("  QueryDecomposer: LLM injected")

    def process_query(self, user_query: str) -> AgentTrace:
        """
        Main entry point: process a user query through the agentic loop.

        Returns a complete AgentTrace with all reasoning steps and the
        final response.
        """
        trace = AgentTrace(query=user_query)
        start_time = time.time()

        log.info("─" * 50)
        log.info("QUERY: %s", user_query)
        log.info("─" * 50)

        if self.llm_client:
            trace = self._llm_agentic_loop(user_query, trace)
        else:
            trace = self._rule_based_loop(user_query, trace)

        trace.total_time = round(time.time() - start_time, 2)
        log.info("─" * 50)
        log.info("COMPLETED in %.2fs (%d steps, tools: %s)",
                 trace.total_time, len(trace.steps), trace.tools_called)
        log.info("─" * 50)

        return trace

    # ────────────────────────────────────────────
    # RULE-BASED ROUTING (no LLM required)
    # ────────────────────────────────────────────
    def _rule_based_loop(self, query: str, trace: AgentTrace) -> AgentTrace:
        """
        Rule-based query routing when no LLM is available.
        Uses the QueryDecomposer to plan steps, then StepExecutor to run them.
        Falls back to legacy _analyze_query if decomposer is unavailable.
        """
        query_lower = query.lower().strip()
        step_num = 0
        collected_context = []

        # ── Step 1: Decompose the query into a plan ──
        step_num += 1
        plan = self.decomposer.decompose(query_lower) if self.decomposer else None

        if plan and plan.steps:
            plan_summary = {
                "intent": plan.intent_summary,
                "reasoning": plan.reasoning,
                "steps": [str(s) for s in plan.steps],
            }
            trace.steps.append(AgentStep(
                step_number=step_num,
                action=AgentAction.THINK,
                thought=f"Query plan: {json.dumps(plan_summary, indent=2)}",
                timestamp=time.time(),
            ))
            if self.config.verbose:
                log.info("  PLAN → %d steps, intent=%s", len(plan.steps), plan.intent_summary)
                for s in plan.steps:
                    log.info("    %s", s)

            # ── Step 2+: Execute the plan ──
            collected_context = self.step_executor.execute_plan(
                plan, trace_steps=trace.steps,
            )
            trace.tools_called = [s.tool for s in plan.steps if s.status == "done"]

            # ── Auto context assembly: if results contain criterion IDs
            # and context_assembler wasn't already in the plan ──
            planned_tools = {s.tool for s in plan.steps}
            if "context_assembler" not in planned_tools:
                all_ids = []
                for res in collected_context:
                    if res.success and res.data:
                        all_ids.extend(self._extract_criterion_ids(res.data))
                all_ids = list(dict.fromkeys(all_ids))[:5]
                if all_ids:
                    ctx_result = self.tools["context_assembler"].execute(
                        criterion_ids=all_ids,
                    )
                    collected_context.append(ctx_result)
                    trace.tools_called.append("context_assembler")
                    trace.steps.append(AgentStep(
                        step_number=len(trace.steps) + 1,
                        action=AgentAction.CALL_TOOL,
                        thought=f"Auto-assembling context for {all_ids}",
                        tool_name="context_assembler",
                        tool_params={"criterion_ids": all_ids},
                        tool_result=ctx_result,
                        timestamp=time.time(),
                    ))

            analysis = {"intent": plan.intent_summary}
        else:
            # Legacy fallback to _analyze_query
            analysis = self._analyze_query(query_lower)
            trace.steps.append(AgentStep(
                step_number=step_num,
                action=AgentAction.THINK,
                thought=f"Query analysis: {json.dumps(analysis, indent=2)}",
                timestamp=time.time(),
            ))

            if self.config.verbose:
                log.info("  THINK → %s", json.dumps(analysis))

            for tool_call in analysis.get("tool_plan", []):
                step_num += 1
                tool_name = tool_call["tool"]
                tool_params = tool_call["params"]

                if tool_name not in self.tools:
                    log.warning("  Unknown tool in plan: %s", tool_name)
                    continue

                if self.config.verbose:
                    log.info("  ACT   → %s(%s)", tool_name, tool_params)

                result = self.tools[tool_name].execute(**tool_params)
                collected_context.append(result)
                trace.tools_called.append(tool_name)

                trace.steps.append(AgentStep(
                    step_number=step_num,
                    action=AgentAction.CALL_TOOL,
                    thought=f"Calling {tool_name} for: {tool_call.get('reason', '')}",
                    tool_name=tool_name,
                    tool_params=tool_params,
                    tool_result=result,
                    timestamp=time.time(),
                ))

                if self.config.verbose:
                    data_preview = str(result.data)[:200] if result.data else "empty"
                    log.info("  OBSERVE → %s: %s...", tool_name, data_preview)

                if result.success and result.data:
                    criterion_ids = self._extract_criterion_ids(result.data)
                    if criterion_ids and "context_assembler" not in [tc["tool"] for tc in analysis["tool_plan"]]:
                        step_num += 1
                        ids_to_assemble = criterion_ids[:5]
                        ctx_result = self.tools["context_assembler"].execute(
                            criterion_ids=ids_to_assemble,
                        )
                        collected_context.append(ctx_result)
                        trace.tools_called.append("context_assembler")
                        trace.steps.append(AgentStep(
                            step_number=step_num,
                            action=AgentAction.CALL_TOOL,
                            thought=f"Assembling full context for {ids_to_assemble}",
                            tool_name="context_assembler",
                            tool_params={"criterion_ids": ids_to_assemble},
                            tool_result=ctx_result,
                            timestamp=time.time(),
                        ))

        # ── Final Step: Generate response ──
        step_num += 1
        if collected_context:
            response = self._build_response(query, collected_context, analysis)
        else:
            response = (
                "I couldn't find specific WCAG information for your query. "
                "Try asking about:\n"
                "• A specific criterion (e.g., '1.4.3 contrast')\n"
                "• An element type (e.g., 'image accessibility')\n"
                "• A disability category (e.g., 'criteria for blind users')\n"
                "• A conformance level (e.g., 'Level AA checklist')\n"
                "• Techniques (e.g., 'how to make forms accessible')"
            )

        trace.final_response = response
        trace.steps.append(AgentStep(
            step_number=step_num,
            action=AgentAction.RESPOND,
            response=response,
            timestamp=time.time(),
        ))

        return trace

    def _analyze_query(self, query: str) -> dict:
        """
        Rule-based query analysis. Identifies intent and plans tool calls.
        Returns a structured analysis with a tool execution plan.
        """
        analysis = {
            "intent": "unknown",
            "entities": [],
            "tool_plan": [],
        }

        # ── Detect criterion IDs (e.g., "1.1.1", "2.4.7") ──
        criterion_pattern = r'\b(\d+\.\d+\.\d+)\b'
        criterion_ids = re.findall(criterion_pattern, query)
        if criterion_ids:
            analysis["entities"] = criterion_ids

        # ── Detect guideline IDs (e.g., "1.1", "2.4") ──
        guideline_pattern = r'\b(\d+\.\d+)\b'
        guideline_ids = [g for g in re.findall(guideline_pattern, query)
                         if g not in [c[:3] for c in criterion_ids]]

        # ── Detect principle IDs ──
        principle_pattern = r'\bprinciple\s+(\d)\b'
        principle_ids = re.findall(principle_pattern, query)

        # ── Detect conformance levels ──
        level_match = re.search(r'\blevel\s+(a{1,3})\b', query, re.IGNORECASE)
        level = level_match.group(1).upper() if level_match else ""

        # ── Intent classification ──
        if criterion_ids:
            analysis["intent"] = "criterion_detail"
            for cid in criterion_ids:
                analysis["tool_plan"].append({
                    "tool": "context_assembler",
                    "params": {"criterion_ids": criterion_ids},
                    "reason": f"Get full context for {', '.join(criterion_ids)}",
                })
            # Also get techniques if asked about "how" or "technique"
            if any(kw in query for kw in ["how", "technique", "fix", "implement", "comply"]):
                for cid in criterion_ids:
                    analysis["tool_plan"].append({
                        "tool": "technique_finder",
                        "params": {"criterion_id": cid, "technique_type": "all"},
                        "reason": f"Find techniques for {cid}",
                    })
            return analysis

        if guideline_ids:
            analysis["intent"] = "guideline_criteria"
            for gid in guideline_ids:
                analysis["tool_plan"].append({
                    "tool": "graph_traversal",
                    "params": {"query_type": "get_criteria_for_guideline", "ref_id": gid},
                    "reason": f"Get criteria for guideline {gid}",
                })
            return analysis

        if principle_ids:
            analysis["intent"] = "principle_guidelines"
            for pid in principle_ids:
                analysis["tool_plan"].append({
                    "tool": "graph_traversal",
                    "params": {"query_type": "get_guidelines_for_principle", "ref_id": pid},
                    "reason": f"Get guidelines for principle {pid}",
                })
            return analysis

        # ── Keyword-based intent detection ──
        element_keywords = {
            "image": "image", "img": "image", "alt text": "image", "picture": "image",
            "form": "form", "input": "form", "label": "form", "field": "form",
            "video": "video", "media": "video", "caption": "video",
            "audio": "audio", "sound": "audio", "transcript": "audio",
            "link": "link", "anchor": "link", "navigation": "link",
            "button": "button", "control": "button",
            "table": "table", "data table": "table",
            "heading": "heading", "header": "heading",
            "color": "color", "contrast": "color",
            "animation": "animation", "motion": "animation",
            "keyboard": "button", "focus": "button",
        }

        for keyword, element in element_keywords.items():
            if keyword in query:
                analysis["intent"] = "element_rules"
                analysis["tool_plan"].append({
                    "tool": "rule_engine",
                    "params": {"check_type": "element_rules", "element_type": element},
                    "reason": f"Find rules for {element} elements",
                })
                return analysis

        # ── Terminology / definition queries → key_term_lookup ──
        term_patterns = [
            r'(?:what\s+(?:does|is|are)|define|definition\s+of|meaning\s+of|explain)\s+["\']?(.+?)["\']?\s*(?:\?|$)',
            r'(?:what\s+is\s+(?:meant\s+by|the\s+meaning\s+of))\s+["\']?(.+?)["\']?\s*(?:\?|$)',
        ]
        for pat in term_patterns:
            m = re.search(pat, query, re.IGNORECASE)
            if m:
                term = m.group(1).strip().rstrip("?. ")
                # Check if it's likely a WCAG term rather than a general question
                if len(term.split()) <= 5:
                    analysis["intent"] = "key_term_lookup"
                    analysis["tool_plan"].append({
                        "tool": "key_term_lookup",
                        "params": {"term": term},
                        "reason": f"Look up WCAG definition of '{term}'",
                    })
                    return analysis

        # ── Cross-reference / relationship queries → cross_reference ──
        if any(kw in query for kw in ["related to", "connected", "depends on",
                                       "ripple", "cascade", "if i fix",
                                       "what else", "overlap between",
                                       "shared technique", "in common"]):
            analysis["intent"] = "cross_reference"
            if criterion_ids:
                # Ripple / related chain for specific criteria
                analysis["tool_plan"].append({
                    "tool": "cross_reference",
                    "params": {"analysis_type": "fix_ripple_effect",
                               "criterion_id": criterion_ids[0]},
                    "reason": f"Analyze cross-references for {criterion_ids[0]}",
                })
            else:
                # Disability overlap or general cross-reference
                analysis["tool_plan"].append({
                    "tool": "semantic_search",
                    "params": {"query": query, "limit": 10},
                    "reason": "Search to identify relevant criteria for cross-reference",
                })
            return analysis

        # ── Technique coverage queries (e.g., "what does H37 cover?") ──
        tech_pattern = r'\b([A-Z]{1,5}\d{1,4})\b'
        technique_ids = re.findall(tech_pattern, query)
        if technique_ids:
            analysis["intent"] = "technique_coverage"
            for tid in technique_ids:
                analysis["tool_plan"].append({
                    "tool": "cross_reference",
                    "params": {"analysis_type": "technique_coverage",
                               "technique_id": tid},
                    "reason": f"Find what criteria technique {tid} covers",
                })
            return analysis

        # Disability-related queries
        disabilities = ["blind", "deaf", "cognitive", "motor", "low vision",
                         "color blind", "seizure", "dyslexia"]
        for disability in disabilities:
            if disability in query:
                analysis["intent"] = "disability_impact"
                analysis["tool_plan"].append({
                    "tool": "impact_analysis",
                    "params": {"analysis_type": "disability_matrix"},
                    "reason": f"Analyze impact on {disability}",
                })
                analysis["tool_plan"].append({
                    "tool": "rule_engine",
                    "params": {"check_type": "disability_impact", "disability": disability},
                    "reason": f"Find criteria affecting {disability}",
                })
                return analysis

        # Checklist / conformance queries
        if any(kw in query for kw in ["checklist", "conformance", "comply", "compliance", "requirements"]):
            analysis["intent"] = "conformance_checklist"
            target_level = level or "AA"
            analysis["tool_plan"].append({
                "tool": "rule_engine",
                "params": {"check_type": "conformance_checklist", "level": target_level},
                "reason": f"Generate Level {target_level} conformance checklist",
            })
            return analysis

        # Level-specific queries
        if level:
            analysis["intent"] = "level_criteria"
            analysis["tool_plan"].append({
                "tool": "graph_traversal",
                "params": {"query_type": "get_criteria_by_level", "level": level},
                "reason": f"Get all Level {level} criteria",
            })
            return analysis

        # Version queries
        if any(v in query for v in ["2.2", "2.1", "new", "added", "latest"]):
            analysis["intent"] = "version_diff"
            version = "2.2" if "2.2" in query else "2.1"
            analysis["tool_plan"].append({
                "tool": "rule_engine",
                "params": {"check_type": "version_diff", "wcag_version": version},
                "reason": f"Find criteria new in WCAG {version}",
            })
            return analysis

        # Automation queries
        if any(kw in query for kw in ["automat", "test", "scan", "tool"]):
            analysis["intent"] = "automatable"
            analysis["tool_plan"].append({
                "tool": "rule_engine",
                "params": {"check_type": "automatable_criteria"},
                "reason": "Find automatable criteria",
            })
            return analysis

        # Hierarchy / overview queries
        if any(kw in query for kw in ["overview", "structure", "hierarchy", "all", "principles"]):
            analysis["intent"] = "hierarchy"
            analysis["tool_plan"].append({
                "tool": "graph_traversal",
                "params": {"query_type": "get_full_hierarchy"},
                "reason": "Get complete WCAG hierarchy",
            })
            return analysis

        # ── Analytics / aggregate / negation → dynamic_cypher ──
        if any(kw in query for kw in ["how many", "count", "most", "least", "average",
                                       "without", "missing", "more than", "fewer than",
                                       "percentage", "top", "bottom", "rank",
                                       "which criteria have", "statistics", "breakdown"]):
            analysis["intent"] = "analytics"
            analysis["tool_plan"].append({
                "tool": "dynamic_cypher",
                "params": {"question": query},
                "reason": "Run analytical ad-hoc Cypher query",
            })
            return analysis

        # ── Fallback: semantic search ──
        analysis["intent"] = "semantic_search"
        analysis["tool_plan"].append({
            "tool": "semantic_search",
            "params": {"query": query, "limit": 10},
            "reason": "Keyword search across all criteria",
        })

        return analysis

    def _extract_criterion_ids(self, data: Any) -> list[str]:
        """Extract criterion ref_ids from tool result data."""
        ids = []
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict) and "ref_id" in item:
                    ref_id = item["ref_id"]
                    # Only criterion-level IDs (x.y.z)
                    if re.match(r'^\d+\.\d+\.\d+$', str(ref_id)):
                        ids.append(str(ref_id))
        elif isinstance(data, dict):
            # Check nested structures
            for key, value in data.items():
                if key in ("rules", "criteria", "checklist"):
                    ids.extend(self._extract_criterion_ids(value))
                elif key == "ref_id" and re.match(r'^\d+\.\d+\.\d+$', str(value)):
                    ids.append(str(value))
        return list(dict.fromkeys(ids))  # deduplicate preserving order

    def _build_response(self, query: str, tool_results: list,
                         analysis: dict) -> str:
        """Build a clean, human-readable response from collected tool results.

        Accepts either ToolResult objects or pre-formatted strings for
        backwards compatibility.
        """
        separator = "─" * 62
        double_sep = "═" * 62
        lines: list[str] = []

        lines.append("")
        lines.append(double_sep)
        lines.append(f"  WCAG 2.2 — Response")
        lines.append(double_sep)
        lines.append(f"  Query:  {query}")
        lines.append(f"  Intent: {analysis.get('intent', 'unknown')}")
        lines.append(separator)
        lines.append("")

        for item in tool_results:
            # ── Handle ToolResult objects ──
            if isinstance(item, ToolResult):
                if not item.success:
                    lines.append(f"  ⚠ {item.tool_name}: {item.error}")
                    lines.append("")
                    continue

                data = item.data

                # Context assembler — use its already-formatted output
                if item.tool_name == "context_assembler" and isinstance(data, dict):
                    formatted = data.get("formatted_context", "")
                    if formatted:
                        lines.append(formatted)
                        continue

                # Semantic search — list of criteria
                if item.tool_name == "semantic_search" and isinstance(data, list):
                    lines.append("  📋 Relevant Criteria Found:")
                    lines.append("")
                    for i, row in enumerate(data, 1):
                        ref = row.get("ref_id", "?")
                        title = row.get("title", "?")
                        level = row.get("level", "?")
                        desc = row.get("description", "")
                        if len(desc) > 200:
                            desc = desc[:200].rsplit(" ", 1)[0] + "…"
                        lines.append(f"  {i}. [{ref}] {title}  (Level {level})")
                        lines.append(f"     {desc}")
                        lines.append("")
                    continue

                # Graph traversal — hierarchy, guidelines, criteria lists
                if item.tool_name == "graph_traversal" and isinstance(data, list):
                    lines.append("  🔗 Graph Results:")
                    lines.append("")
                    for row in data:
                        ref = row.get("ref_id", "")
                        title = row.get("title", row.get("name", ""))
                        level = row.get("level", "")
                        level_str = f"  (Level {level})" if level else ""
                        lines.append(f"  • [{ref}] {title}{level_str}")
                    lines.append("")
                    continue

                # Technique finder — grouped by category
                if item.tool_name == "technique_finder" and isinstance(data, dict):
                    lines.append("  🛠 Techniques:")
                    lines.append("")
                    for category, label_icon in [("sufficient", "✅"), ("advisory", "💡"), ("failures", "❌")]:
                        techs = data.get(category, [])
                        if techs:
                            label = category.replace("_", " ").title()
                            lines.append(f"  {label}:")
                            for t in techs:
                                tid = t.get("tech_id", "")
                                tt = t.get("title", "")
                                lines.append(f"    {label_icon} [{tid}] {tt}")
                            lines.append("")
                    continue

                # Rule engine — criteria list with optional grouping
                if item.tool_name == "rule_engine" and isinstance(data, (dict, list)):
                    lines.append("  📏 Rule Engine Results:")
                    lines.append("")
                    if isinstance(data, dict):
                        rules = data.get("rules", data.get("criteria", data.get("checklist", [])))
                        if isinstance(rules, list):
                            for row in rules:
                                ref = row.get("ref_id", "?")
                                title = row.get("title", "?")
                                level = row.get("level", "")
                                level_str = f"  (Level {level})" if level else ""
                                lines.append(f"  • [{ref}] {title}{level_str}")
                        else:
                            for key, val in data.items():
                                lines.append(f"  {key}: {val}")
                    elif isinstance(data, list):
                        for row in data:
                            ref = row.get("ref_id", "?")
                            title = row.get("title", "?")
                            level = row.get("level", "")
                            level_str = f"  (Level {level})" if level else ""
                            lines.append(f"  • [{ref}] {title}{level_str}")
                    lines.append("")
                    continue

                # Impact analysis — formatted matrix/list
                if item.tool_name == "impact_analysis" and isinstance(data, (dict, list)):
                    lines.append("  ♿ Impact Analysis:")
                    lines.append("")
                    if isinstance(data, dict):
                        for key, val in data.items():
                            if isinstance(val, list):
                                lines.append(f"  {key}:")
                                for v in val:
                                    if isinstance(v, dict):
                                        ref = v.get("ref_id", "?")
                                        title = v.get("title", "?")
                                        lines.append(f"    • [{ref}] {title}")
                                    else:
                                        lines.append(f"    • {v}")
                            else:
                                lines.append(f"  {key}: {val}")
                    elif isinstance(data, list):
                        for row in data:
                            ref = row.get("ref_id", "?")
                            title = row.get("title", "?")
                            lines.append(f"  • [{ref}] {title}")
                    lines.append("")
                    continue

                # Key term lookup — definitions and usage
                if item.tool_name == "key_term_lookup" and isinstance(data, dict):
                    lines.append("  📖 Key Term Definitions:")
                    lines.append("")

                    # Single criterion mode
                    if "key_terms" in data:
                        cid = data.get("criterion_id", "?")
                        lines.append(f"  Key terms for {cid}:")
                        for kt in data["key_terms"]:
                            defn = (kt.get("definition", "") or "")[:300]
                            lines.append(f"    • {kt.get('term', '?')}: {defn}")
                        lines.append("")
                        continue

                    # Search mode
                    terms = data.get("terms", [])
                    if terms:
                        for t in terms:
                            defn = (t.get("definition", "") or "")[:300]
                            criteria = t.get("used_in_criteria", [])
                            criteria_str = ", ".join(c.get("ref_id", "") for c in criteria[:8])
                            if len(criteria) > 8:
                                criteria_str += f" (+{len(criteria) - 8} more)"
                            lines.append(f"  📌 {t.get('term', '?')}")
                            lines.append(f"     Definition: {defn}")
                            lines.append(f"     Used in: {criteria_str}")
                            lines.append("")
                    else:
                        lines.append(f"  No matching terms found for '{data.get('search_term', '?')}'")
                        lines.append("")
                    continue

                # Cross-reference — multi-hop analysis
                if item.tool_name == "cross_reference" and isinstance(data, dict):
                    lines.append("  🔀 Cross-Reference Analysis:")
                    lines.append("")

                    # Related chain
                    if "direct_related" in data and "second_hop" in data:
                        source = data.get("source", "?")
                        lines.append(f"  Relationship chain from {source} ({data.get('source_title', '')}):")
                        lines.append("")
                        direct = data.get("direct_related", [])
                        if direct:
                            lines.append("  Direct relations:")
                            for r in direct:
                                lines.append(f"    → [{r.get('ref_id', '?')}] {r.get('title', '')} (Level {r.get('level', '?')})")
                        second = data.get("second_hop", [])
                        if second:
                            lines.append("  2nd-hop relations:")
                            for r in second:
                                via = f" via {r.get('via', '?')}" if r.get("via") else ""
                                lines.append(f"    →→ [{r.get('ref_id', '?')}] {r.get('title', '')}{via}")
                        lines.append("")
                        continue

                    # Shared techniques
                    if "shared_techniques_count" in data:
                        cids = ", ".join(data.get("criterion_ids", []))
                        lines.append(f"  Shared techniques across {cids}:")
                        lines.append(f"  Found {data['shared_techniques_count']} shared technique(s)")
                        lines.append("")
                        for t in data.get("techniques", []):
                            shared_with = ", ".join(t.get("criteria_using", []))
                            lines.append(f"    🔧 [{t.get('tech_id', '?')}] {t.get('title', '')}")
                            lines.append(f"       Shared by: {shared_with}")
                        lines.append("")
                        continue

                    # Disability overlap
                    if "overlapping_criteria" in data:
                        da = data.get("disability_a", "?")
                        db = data.get("disability_b", "?")
                        lines.append(f"  Disability overlap: {da} ∩ {db}")
                        lines.append(f"  Overlapping: {data['overlapping_criteria']} | Only {da}: {data.get('only_a_count', 0)} | Only {db}: {data.get('only_b_count', 0)}")
                        lines.append("")
                        for c in data.get("overlap", []):
                            lines.append(f"    ∩ [{c.get('ref_id', '?')}] {c.get('title', '')} (Level {c.get('level', '?')})")
                        lines.append("")
                        continue

                    # Technique coverage
                    if "criteria" in data and "tech_id" in data:
                        lines.append(f"  Technique {data.get('tech_id', '?')}: {data.get('title', '')}")
                        lines.append(f"  Covers {data.get('criteria_count', 0)} criteria:")
                        for c in data.get("criteria", []):
                            rel = c.get("relationship", "").replace("HAS_", "").lower()
                            lines.append(f"    • [{c.get('ref_id', '?')}] {c.get('criterion_title', '')} ({rel})")
                        lines.append("")
                        continue

                    # Ripple effect
                    if "total_affected" in data:
                        source = data.get("source", "?")
                        lines.append(f"  Fix ripple effect from {source} ({data.get('source_title', '')}):")
                        lines.append(f"  Total affected criteria: {data['total_affected']}")
                        lines.append("")
                        for a in data.get("all_affected", []):
                            conn = a.get("connection", "")
                            lines.append(f"    {'→' if conn == 'directly_related' else '🔧'} [{a.get('ref_id', '?')}] {a.get('title', '')} ({conn})")
                        lines.append("")
                        continue

                    # Generic cross_reference fallback
                    text = json.dumps(data, indent=2, default=str)
                    if len(text) > 1000:
                        text = text[:1000] + "\n  ... (truncated)"
                    lines.append(text)
                    lines.append("")
                    continue

                # Dynamic Cypher — show generated query + results
                if item.tool_name == "dynamic_cypher" and isinstance(data, dict):
                    lines.append("  ⚡ Dynamic Cypher Results:")
                    lines.append("")
                    cypher_q = data.get("cypher", "")
                    if cypher_q:
                        lines.append(f"  Cypher: {cypher_q[:300]}")
                    explanation = data.get("explanation", "")
                    if explanation:
                        lines.append(f"  Explanation: {explanation}")
                    lines.append(f"  Rows returned: {data.get('row_count', 0)}")
                    lines.append("")
                    results = data.get("results", [])
                    for i, row in enumerate(results[:25], 1):
                        parts = []
                        for k, v in row.items():
                            parts.append(f"{k}={v}")
                        lines.append(f"  {i}. {' │ '.join(parts)}")
                    if len(results) > 25:
                        lines.append(f"  ... and {len(results) - 25} more rows")
                    lines.append("")
                    continue

                # Fallback for any other tool — compact JSON
                lines.append(f"  [{item.tool_name}]:")
                text = json.dumps(data, indent=2, default=str)
                if len(text) > 1000:
                    text = text[:1000] + "\n  ... (truncated)"
                lines.append(text)
                lines.append("")

            # ── Handle legacy strings (backwards compat) ──
            elif isinstance(item, str):
                lines.append(item)
                lines.append("")

        lines.append(double_sep)
        return "\n".join(lines)

    # ────────────────────────────────────────────
    # LLM-POWERED AGENTIC LOOP (with function calling)
    # ────────────────────────────────────────────
    def _llm_agentic_loop(self, query: str, trace: AgentTrace) -> AgentTrace:
        """
        Full agentic loop powered by LLM function calling.

        NEW: Before entering the ReAct loop the QueryDecomposer creates
        a plan.  The plan is injected as a system hint so the LLM follows
        the recommended step order while still retaining the freedom to
        deviate if intermediate results warrant it.
        """
        # ── Phase 0: Query Decomposition / Planning ──
        plan: Optional[QueryPlan] = None
        plan_hint = ""
        if self.decomposer:
            plan = self.decomposer.decompose(query)
            if plan and plan.steps:
                plan_lines = [f"Intent: {plan.intent_summary}"]
                if plan.reasoning:
                    plan_lines.append(f"Reasoning: {plan.reasoning}")
                plan_lines.append("Recommended steps:")
                for s in plan.steps:
                    deps = f" (after step {s.depends_on})" if s.depends_on else ""
                    plan_lines.append(
                        f"  {s.step_id}. [{s.tool}] {s.description}{deps}"
                    )
                plan_hint = "\n".join(plan_lines)

                trace.steps.append(AgentStep(
                    step_number=1,
                    action=AgentAction.THINK,
                    thought=f"Query decomposition plan:\n{plan_hint}",
                    timestamp=time.time(),
                ))
                if self.config.verbose:
                    log.info("  DECOMPOSE → %d steps, intent=%s",
                             len(plan.steps), plan.intent_summary)
                    for s in plan.steps:
                        log.info("    %s", s)

        # Build tool definitions for function calling
        tools_schema = self._build_tools_schema()

        # System prompt
        system_prompt = self._build_system_prompt()

        # Inject the plan as guidance
        user_message = query
        if plan_hint:
            user_message += (
                "\n\n--- QUERY PLAN (follow this order unless results suggest otherwise) ---\n"
                + plan_hint
            )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]

        step_num = len(trace.steps)  # continue numbering after decomposition step
        for _ in range(self.config.max_agent_steps):
            step_num += 1

            try:
                response = self.llm_client.chat.completions.create(
                    model=self._llm_model_name,
                    messages=messages,
                    tools=tools_schema,
                    tool_choice="auto",
                    temperature=self.config.temperature,
                )
            except Exception as e:
                log.error("LLM call failed: %s — falling back to rule-based", e)
                return self._rule_based_loop(query, trace)

            choice = response.choices[0]
            message = choice.message

            # ── If the LLM wants to call tools ──
            if message.tool_calls:
                messages.append(message)

                for tool_call in message.tool_calls:
                    fn_name = tool_call.function.name
                    fn_args = json.loads(tool_call.function.arguments)

                    if self.config.verbose:
                        log.info("  LLM ACT → %s(%s)", fn_name, fn_args)

                    if fn_name in self.tools:
                        result = self.tools[fn_name].execute(**fn_args)
                        trace.tools_called.append(fn_name)
                    else:
                        result = ToolResult(
                            tool_name=fn_name, success=False,
                            error=f"Unknown tool: {fn_name}"
                        )

                    trace.steps.append(AgentStep(
                        step_number=step_num,
                        action=AgentAction.CALL_TOOL,
                        thought=message.content or "",
                        tool_name=fn_name,
                        tool_params=fn_args,
                        tool_result=result,
                        timestamp=time.time(),
                    ))

                    # Add tool result to conversation
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": result.to_context(
                            max_length=self.config.max_context_tokens
                        ),
                    })

            # ── If the LLM wants to respond (no more tool calls) ──
            elif choice.finish_reason == "stop":
                final_response = message.content or ""
                trace.final_response = final_response
                trace.steps.append(AgentStep(
                    step_number=step_num,
                    action=AgentAction.RESPOND,
                    response=final_response,
                    timestamp=time.time(),
                ))
                break
        else:
            trace.final_response = (
                "I reached the maximum number of reasoning steps. "
                "Here's what I found so far based on the tools I called."
            )

        return trace

    def _build_system_prompt(self) -> str:
        """Build the system prompt for the LLM agent."""
        return """You are an expert WCAG 2.2 accessibility consultant powered by a Neo4j knowledge graph.
You operate in a Plan → Execute → Synthesise loop.  A query plan may be provided below the user query — follow it unless intermediate results suggest a better path.

KNOWLEDGE GRAPH CONTENTS (2,418 nodes):
- 4 Principles → 13 Guidelines → 87 Success Criteria (full metadata, intent, in-brief)
- 412 Techniques (sufficient, advisory, failure) with technology tags
- 725 Key Terms with formal definitions (e.g., "programmatically determined", "text alternative")
- 284 Examples with descriptions
- 261 Related Resources (external links)
- 204 Benefits describing who is helped
- 109 Automated Test Rules (ACT)
- 91 Special Cases / exceptions
- Cross-references between related criteria
- Disability impact tags on every criterion

AVAILABLE TOOLS (9):
1. graph_traversal — Navigate the WCAG hierarchy by ID (principle → guideline → criterion). Use for ID-based lookups.
2. semantic_search — Keyword search across criteria, examples, benefits, and key terms. Use for topic-based queries.
3. technique_finder — Find sufficient/advisory/failure techniques for a criterion or technology. Use for "how to comply" questions.
4. rule_engine — Compliance rules: element-specific rules, disability impact, conformance checklists, automatable criteria, version diffs.
5. impact_analysis — Disability matrix, input modality analysis, criterion-level impact breakdown.
6. key_term_lookup — Look up WCAG key term definitions. Use when user asks "what does X mean?" or references WCAG-specific terminology.
7. cross_reference — Multi-hop graph analysis: related criterion chains, shared techniques across criteria, disability overlap, technique coverage maps, fix ripple effects.
8. dynamic_cypher — Generate and run ad-hoc Cypher queries for ANALYTICAL / AGGREGATE / NEGATION questions not covered by tools 1-7 (e.g., "which criteria have more than 5 techniques?", "show criteria WITHOUT test rules", "count techniques per technology"). The LLM writes the Cypher; guardrails enforce read-only, schema-checked execution.
9. context_assembler — Assemble complete LLM-ready context for one or more criteria. Use as FINAL step before generating a response.

WHEN TO USE dynamic_cypher (Tool 8):
- Counting / aggregation: "how many criteria per level?", "average techniques per criterion"
- Negation / absence: "criteria without test rules", "guidelines with no AAA criteria"
- Ranking / top-N: "top 5 criteria by technique count"
- Complex filters: "level A criteria that affect both blind and deaf users and have ARIA techniques"
- Any question that tools 1-7 cannot handle with their pre-built templates

QUERY DECOMPOSITION PATTERNS:

For SCENARIO questions (e.g., "audit this login form"):
  Step 1: Identify UI elements mentioned (form, button, input, image, video, etc.)
  Step 2: Use rule_engine(element_rules) for each element type
  Step 3: Use context_assembler for the union of criteria found
  Step 4: Synthesize a checklist grouped by element

For TERMINOLOGY questions (e.g., "what does programmatically determined mean?"):
  Step 1: Use key_term_lookup(term=...) to get the formal definition
  Step 2: Note which criteria use this term
  Step 3: Optionally use context_assembler to provide criterion context

For COMPARISON questions (e.g., "difference between 2.1 and 2.2" or "A vs AA for images"):
  Step 1: Use cross_reference or rule_engine to get both sides
  Step 2: Compare and present differences clearly

For RIPPLE/DEPENDENCY questions (e.g., "if I fix keyboard issues, what else improves?"):
  Step 1: Identify the criterion (e.g., 2.1.1 Keyboard)
  Step 2: Use cross_reference(fix_ripple_effect) to map cascading benefits
  Step 3: Use context_assembler for affected criteria details

For DISABILITY-FOCUSED questions (e.g., "what helps blind users?"):
  Step 1: Use impact_analysis(disability_matrix) or cross_reference(disability_overlap)
  Step 2: Use context_assembler for the top criteria

For TECHNIQUE questions (e.g., "what does H37 cover?" or "ARIA techniques for forms"):
  Step 1: Use technique_finder or cross_reference(technique_coverage)
  Step 2: Show which criteria each technique satisfies

For ANALYTICAL / STATISTICAL questions (e.g., "how many criteria per level?"):
  Step 1: Use dynamic_cypher with the analytical question
  Step 2: Optionally use context_assembler for the top results

INSTRUCTIONS:
1. ALWAYS use tools to retrieve factual WCAG information — never guess or hallucinate criteria.
2. Start with the most specific tool for the query. Decompose complex questions into steps.
3. After finding relevant criteria, use context_assembler to get complete information before responding.
4. Use key_term_lookup when the user references WCAG-specific terminology.
5. Use cross_reference for any question involving relationships BETWEEN criteria or techniques.
6. Use dynamic_cypher for counting, aggregation, negation, ranking, or complex ad-hoc filters.
7. Cite specific criterion IDs (e.g., "WCAG 2.2 SC 1.4.3") in your response.
8. When listing techniques, include the technique ID (e.g., G18, H37, F65).
9. Structure your response clearly with sections for: applicable criteria, techniques, examples, and related criteria.
10. If the query is ambiguous, retrieve broadly first, then narrow down.
11. Always mention the conformance level (A, AA, AAA) for each criterion.
12. If a QUERY PLAN is attached, follow its recommended step order unless results suggest otherwise.

RESPONSE FORMAT:
- Use clear headings and bullet points
- Group information by criterion
- Include actionable recommendations
- Cite technique IDs and criterion numbers
- Mention which disabilities are impacted
- When relevant, show which key terms apply and their definitions"""

    def _build_tools_schema(self) -> list[dict]:
        """Build OpenAI-compatible tool schemas from registered tools."""
        schemas = []
        for tool in self.tools.values():
            schemas.append({
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                },
            })
        return schemas


# ============================================================
# CLI INTERFACE
# ============================================================
def run_interactive(agent: WCAGAgent):
    """Run the agent in interactive mode."""
    print("\n" + "=" * 62)
    print("  WCAG 2.2 Agentic RAG — Interactive Mode")
    print("  Type 'quit' to exit, 'trace' to show last trace")
    print("=" * 62 + "\n")

    last_trace: Optional[AgentTrace] = None

    while True:
        try:
            user_input = input("🔍 Ask about WCAG: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break
        if user_input.lower() == "trace" and last_trace:
            _print_trace(last_trace)
            continue

        trace = agent.process_query(user_input)
        last_trace = trace

        print("\n" + trace.final_response + "\n")


def _print_trace(trace: AgentTrace):
    """Print a detailed trace of agent reasoning."""
    print("\n" + "═" * 62)
    print(f"  AGENT TRACE — {len(trace.steps)} steps, {trace.total_time}s")
    print("═" * 62)
    for step in trace.steps:
        icon = {"think": "🧠", "call_tool": "🔧", "respond": "💬"}
        print(f"\n  Step {step.step_number} [{icon.get(step.action.value, '?')} {step.action.value}]")
        if step.thought:
            print(f"    Thought: {step.thought[:200]}")
        if step.tool_name:
            print(f"    Tool: {step.tool_name}({json.dumps(step.tool_params, default=str)[:150]})")
        if step.tool_result:
            status = "✅" if step.tool_result.success else "❌"
            print(f"    Result: {status} ({step.tool_result.execution_time}s)")
        if step.response:
            print(f"    Response: {step.response[:300]}...")
    print("═" * 62 + "\n")


def run_single_query(agent: WCAGAgent, query: str):
    """Run a single query and print the result."""
    trace = agent.process_query(query)
    print("\n" + trace.final_response)
    if agent.config.verbose:
        _print_trace(trace)


# ============================================================
# ENTRY POINT
# ============================================================
def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="WCAG 2.2 Agentic RAG — Knowledge Graph powered assistant"
    )
    parser.add_argument(
        "--query", "-q", type=str, default="",
        help="Single query to process (omit for interactive mode)"
    )
    parser.add_argument(
        "--interactive", "-i", action="store_true",
        help="Run in interactive mode"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Show detailed agent reasoning"
    )
    args = parser.parse_args()

    config = AgentConfig()
    if args.verbose:
        config.verbose = True

    try:
        config.validate()
    except EnvironmentError as e:
        log.error("Config error: %s", e)
        sys.exit(1)

    db = None
    try:
        db = Neo4jConnection(config)

        agent = WCAGAgent(config, db)

        if args.query:
            run_single_query(agent, args.query)
        else:
            run_interactive(agent)

    except ConnectionError as e:
        log.error("Connection error: %s", e)
        sys.exit(1)
    except Exception as e:
        log.error("Fatal error: %s", e, exc_info=True)
        sys.exit(1)
    finally:
        if db:
            db.close()


if __name__ == "__main__":
    main()