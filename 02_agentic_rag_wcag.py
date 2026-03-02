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

        # Build WHERE clause for keyword matching
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

        where_clause = " OR ".join(where_conditions)
        params = {f"kw{i}": kw for i, kw in enumerate(keywords)}
        params["limit"] = limit

        if level_filter:
            where_clause = f"({where_clause}) AND c.level = $level_filter"
            params["level_filter"] = level_filter

        cypher = f"""
            MATCH (c:WCAGCriterion)
            WHERE {where_clause}
            OPTIONAL MATCH (c)-[:PART_OF]->(g:WCAGGuideline)-[:PART_OF]->(p:WCAGPrinciple)
            RETURN c.ref_id AS ref_id, c.title AS title,
                   c.level AS level, c.description AS description,
                   c.intent AS intent, c.wcag_version AS wcag_version,
                   c.in_brief_goal AS goal,
                   c.in_brief_what_to_do AS what_to_do,
                   g.title AS guideline, p.title AS principle
            ORDER BY c.ref_id
            LIMIT $limit
        """

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
# TOOL 6: CONTEXT ASSEMBLER
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

            sections.append(section)

        return "\n".join(sections)


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
            ContextAssemblerTool,
        ]
        for cls in tool_classes:
            tool = cls(self.db)
            self.tools[tool.name] = tool
            log.info("  Registered tool: %s", tool.name)

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
        Analyzes the query structure and routes to appropriate tools.
        """
        query_lower = query.lower().strip()
        step_num = 0
        collected_context = []

        # ── Step 1: Query Analysis ──
        step_num += 1
        analysis = self._analyze_query(query_lower)
        trace.steps.append(AgentStep(
            step_number=step_num,
            action=AgentAction.THINK,
            thought=f"Query analysis: {json.dumps(analysis, indent=2)}",
            timestamp=time.time(),
        ))

        if self.config.verbose:
            log.info("  THINK → %s", json.dumps(analysis))

        # ── Step 2+: Execute planned tool calls ──
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

            # If we found specific criterion IDs, assemble full context
            if result.success and result.data:
                criterion_ids = self._extract_criterion_ids(result.data)
                if criterion_ids and "context_assembler" not in [tc["tool"] for tc in analysis["tool_plan"]]:
                    step_num += 1
                    # Limit to top 5 to avoid context overflow
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
        The LLM decides which tools to call and when to stop.
        """
        # Build tool definitions for function calling
        tools_schema = self._build_tools_schema()

        # System prompt
        system_prompt = self._build_system_prompt()

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query},
        ]

        step_num = 0
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

You have access to a comprehensive WCAG 2.2 knowledge graph containing:
- 4 Principles (Perceivable, Operable, Understandable, Robust)
- 13 Guidelines
- 86+ Success Criteria with full metadata
- Sufficient, Advisory, and Failure Techniques
- Test Rules for automated testing
- Examples, Benefits, and Disability Impact data
- Cross-references between related criteria

INSTRUCTIONS:
1. ALWAYS use tools to retrieve factual WCAG information — never guess or hallucinate criteria.
2. Start with the most specific tool for the query. Use semantic_search for topic queries, graph_traversal for ID-based queries, rule_engine for compliance questions.
3. After finding relevant criteria, use context_assembler to get complete information before responding.
4. Use technique_finder when the user asks HOW to comply.
5. Use impact_analysis when the user asks about disability categories or input modalities.
6. Cite specific criterion IDs (e.g., "WCAG 2.2 SC 1.4.3") in your response.
7. When listing techniques, include the technique ID (e.g., G18, H37, F65).
8. Structure your response clearly with sections for: applicable criteria, techniques, examples, and related criteria.
9. If the query is ambiguous, retrieve broadly first, then narrow down.
10. Always mention the conformance level (A, AA, AAA) for each criterion.

RESPONSE FORMAT:
- Use clear headings and bullet points
- Group information by criterion
- Include actionable recommendations
- Cite technique IDs and criterion numbers
- Mention which disabilities are impacted"""

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