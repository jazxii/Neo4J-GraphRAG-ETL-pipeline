"""
Pipeline Step 1: WCAG 2.2 Foundation — ETL into Neo4j Knowledge Graph.

A proper ETL pipeline with clearly separated phases:

  EXTRACT  → Read and validate WCAG 2.2 JSON source data (base or enriched)
  TRANSFORM→ Normalize, enrich, and shape records for graph ingestion
  LOAD     → Batch-write nodes and relationships into Neo4j
  VALIDATE → Verify graph integrity post-load

This runs FIRST — before any JIRA data.
Creates the authoritative WCAG knowledge layer that all bugs
will link to via :VIOLATES relationships.

Supports BOTH the base JSON (wcag_22_guidelines.json) and the enriched
JSON (wcag_22_guidelines_enriched.json) produced by 00_enrich_wcag_json.py.
When enriched data is present, additional nodes and relationships are created.

Graph Schema:
  Nodes : WCAGPrinciple · WCAGGuideline · WCAGCriterion
          ConformanceLevel · WCAGSpecialCase · WCAGNote · WCAGReference
          WCAGTechnique · WCAGTestRule · WCAGExample · WCAGBenefit
          WCAGKeyTerm · WCAGRelatedResource
  Edges : PART_OF · HAS_LEVEL · HAS_SPECIAL_CASE · HAS_NOTE
          HAS_REFERENCE · RELATED_CRITERION · RELATED_GUIDELINE
          HAS_TECHNIQUE · HAS_FAILURE · HAS_ADVISORY_TECHNIQUE
          HAS_TEST_RULE · HAS_EXAMPLE · HAS_BENEFIT · IMPACTS_DISABILITY
          HAS_KEY_TERM · HAS_RELATED_RESOURCE

Usage:
  python 01_pipeline_wcag_foundation.py
"""

from neo4j import GraphDatabase
import json
import sys
import os
import time
import logging
from dataclasses import dataclass, field
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# ============================================================
# LOGGING CONFIGURATION
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-7s │ %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("wcag_etl")


# ============================================================
# PIPELINE CONFIGURATION
# ============================================================
@dataclass
class PipelineConfig:
    """Central, immutable configuration for the pipeline run."""
    neo4j_uri: str = os.getenv("NEO4J_URI", "")
    neo4j_user: str = os.getenv("NEO4J_USER", "")
    neo4j_password: str = os.getenv("NEO4J_PASSWORD", "")
    wcag_json_file: str = os.getenv("WCAG_JSON_FILE", "wcag_22_guidelines.json")
    batch_size: int = 100  # Cypher UNWIND batch size

    def validate(self):
        """Fail fast if required config is missing."""
        missing = []
        if not self.neo4j_uri:
            missing.append("NEO4J_URI")
        if not self.neo4j_user:
            missing.append("NEO4J_USER")
        if not self.neo4j_password:
            missing.append("NEO4J_PASSWORD")
        if missing:
            raise EnvironmentError(
                f"Missing required environment variables: {', '.join(missing)}. "
                f"Check your .env file."
            )


# ============================================================
# PIPELINE METRICS
# ============================================================
@dataclass
class PipelineMetrics:
    """Track counts and timing across all phases."""
    principles: int = 0
    guidelines: int = 0
    criteria: int = 0
    special_cases: int = 0
    notes: int = 0
    references: int = 0
    cross_refs: int = 0
    techniques: int = 0
    test_rules: int = 0
    examples: int = 0
    benefits: int = 0
    key_terms: int = 0
    related_resources: int = 0
    enriched: bool = False        # True when enriched JSON is detected
    phase_times: dict = field(default_factory=dict)

    def summary(self) -> dict:
        return {
            k: v for k, v in self.__dict__.items()
            if k != "phase_times"
        }


# ============================================================
# NEO4J SESSION HELPERS
# ============================================================
class Neo4jConnection:
    """Managed Neo4j driver with batch-write support."""

    def __init__(self, config: PipelineConfig):
        self.driver = GraphDatabase.driver(
            config.neo4j_uri,
            auth=(config.neo4j_user, config.neo4j_password),
        )
        self._verify_connectivity()

    def _verify_connectivity(self):
        """Fail fast if Neo4j is unreachable."""
        try:
            self.driver.verify_connectivity()
            log.info("Neo4j connectivity verified")
        except Exception as e:
            raise ConnectionError(f"Cannot reach Neo4j: {e}") from e

    def read(self, query: str, params: dict | None = None) -> list[dict]:
        with self.driver.session() as session:
            return [r.data() for r in session.run(query, params or {})]

    def write(self, query: str, params: dict | None = None):
        """Execute a single write query inside an explicit transaction."""
        with self.driver.session() as session:
            session.execute_write(lambda tx: tx.run(query, params or {}))

    def batch_write(self, query: str, rows: list[dict]):
        """Write many rows in a single UNWIND transaction (much faster)."""
        if not rows:
            return
        with self.driver.session() as session:
            session.execute_write(
                lambda tx: tx.run(query, {"rows": rows})
            )

    def close(self):
        self.driver.close()


# ================================================================
#  PHASE 0 — PRE-FLIGHT: Clean slate + schema constraints
# ================================================================
def phase_preflight(db: Neo4jConnection):
    """Prepare the database: full clean slate + schema constraints."""
    log.info("PHASE 0 ▸ Pre-flight checks")

    # ── Full wipe of ALL WCAG nodes (clean reload with new enriched data) ──
    all_wcag_labels = [
        "WCAGSpecialCase", "WCAGNote", "WCAGReference",
        "WCAGTechnique", "WCAGTestRule", "WCAGExample", "WCAGBenefit",
        "WCAGKeyTerm", "WCAGRelatedResource",
        "WCAGCriterion", "WCAGGuideline", "WCAGPrinciple",
        "ConformanceLevel",
    ]
    for label in all_wcag_labels:
        db.write(f"MATCH (n:{label}) DETACH DELETE n")
    log.info("  Wiped ALL WCAG nodes for clean reload")

    # ── Constraints & indexes ──
    schema_statements = [
        "CREATE CONSTRAINT IF NOT EXISTS FOR (p:WCAGPrinciple)    REQUIRE p.ref_id IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (g:WCAGGuideline)    REQUIRE g.ref_id IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (c:WCAGCriterion)    REQUIRE c.ref_id IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (cl:ConformanceLevel) REQUIRE cl.name  IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (t:WCAGTechnique)    REQUIRE t.tech_id IS UNIQUE",
        "CREATE INDEX IF NOT EXISTS FOR (c:WCAGCriterion) ON (c.code)",
        "CREATE INDEX IF NOT EXISTS FOR (c:WCAGCriterion) ON (c.title)",
        "CREATE INDEX IF NOT EXISTS FOR (c:WCAGCriterion) ON (c.level)",
        "CREATE INDEX IF NOT EXISTS FOR (c:WCAGCriterion) ON (c.wcag_version)",
        "CREATE INDEX IF NOT EXISTS FOR (c:WCAGCriterion) ON (c.automatable)",
        "CREATE INDEX IF NOT EXISTS FOR (g:WCAGGuideline) ON (g.title)",
        "CREATE INDEX IF NOT EXISTS FOR (p:WCAGPrinciple) ON (p.name)",
        "CREATE INDEX IF NOT EXISTS FOR (t:WCAGTechnique) ON (t.technology)",
        "CREATE INDEX IF NOT EXISTS FOR (t:WCAGTechnique) ON (t.category)",
        "CREATE INDEX IF NOT EXISTS FOR (tr:WCAGTestRule) ON (tr.title)",
        "CREATE INDEX IF NOT EXISTS FOR (kt:WCAGKeyTerm) ON (kt.term)",
        "CREATE INDEX IF NOT EXISTS FOR (rr:WCAGRelatedResource) ON (rr.url)",
    ]
    for stmt in schema_statements:
        try:
            db.write(stmt)
        except Exception as e:
            if "already exists" not in str(e).lower():
                log.warning("  Schema issue: %s", e)
    log.info("  Constraints and indexes ready")


# ================================================================
#  PHASE 1 — EXTRACT: Read + validate source JSON
# ================================================================
def phase_extract(config: PipelineConfig) -> list[dict]:
    """Read the WCAG JSON file and validate its structure."""
    log.info("PHASE 1 ▸ Extract — reading %s", config.wcag_json_file)

    # ── Read file ──
    if not os.path.isfile(config.wcag_json_file):
        raise FileNotFoundError(
            f"Source file not found: {config.wcag_json_file}. "
            f"Place your WCAG 2.2 JSON in the project directory."
        )

    with open(config.wcag_json_file, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    # ── Structural validation ──
    if not isinstance(raw_data, list) or len(raw_data) == 0:
        raise ValueError("WCAG JSON must be a non-empty array of principles")

    required_principle_keys = {"ref_id", "title", "description", "url", "guidelines"}
    required_guideline_keys = {"ref_id", "title", "description", "url", "success_criteria"}
    required_criterion_keys = {"ref_id", "title", "description", "url", "level"}

    for p_idx, principle in enumerate(raw_data):
        missing = required_principle_keys - set(principle.keys())
        if missing:
            raise ValueError(f"Principle [{p_idx}] missing keys: {missing}")

        for g_idx, guideline in enumerate(principle.get("guidelines", [])):
            missing = required_guideline_keys - set(guideline.keys())
            if missing:
                raise ValueError(
                    f"Guideline [{p_idx}][{g_idx}] ({guideline.get('ref_id', '?')}) "
                    f"missing keys: {missing}"
                )

            for sc_idx, sc in enumerate(guideline.get("success_criteria", [])):
                missing = required_criterion_keys - set(sc.keys())
                if missing:
                    raise ValueError(
                        f"Criterion [{p_idx}][{g_idx}][{sc_idx}] "
                        f"({sc.get('ref_id', '?')}) missing keys: {missing}"
                    )
                if sc["level"] not in ("A", "AA", "AAA"):
                    raise ValueError(
                        f"Criterion {sc['ref_id']} has invalid level: {sc['level']}"
                    )

    log.info("  Validated %d principles — structure OK", len(raw_data))
    return raw_data


# ================================================================
#  PHASE 2 — TRANSFORM: Flatten & normalize into batch-ready records
# ================================================================
@dataclass
class TransformedData:
    """All records normalized and ready for batch loading."""
    conformance_levels: list[dict]
    principles: list[dict]
    guidelines: list[dict]
    criteria: list[dict]
    special_cases: list[dict]
    notes: list[dict]
    references: list[dict]        # node records
    guideline_refs: list[dict]    # guideline → reference edges
    criterion_refs: list[dict]    # criterion → reference edges
    cross_refs: list[dict]        # criterion ↔ criterion/guideline edges
    # ── Enriched data (empty if using base JSON) ──
    techniques: list[dict]        # WCAGTechnique nodes
    technique_edges: list[dict]   # criterion → technique edges
    test_rules: list[dict]        # WCAGTestRule nodes + edges
    examples: list[dict]          # WCAGExample nodes + edges
    benefits: list[dict]          # WCAGBenefit nodes + edges
    enriched_cross_refs: list[dict]  # dynamically discovered SC ↔ SC refs
    key_terms: list[dict]         # WCAGKeyTerm nodes + edges
    related_resources: list[dict] # WCAGRelatedResource nodes + edges


def phase_transform(raw_data: list[dict], metrics: PipelineMetrics) -> TransformedData:
    """Normalize raw JSON into flat, batch-ready record lists."""
    log.info("PHASE 2 ▸ Transform — normalizing records")

    # ── Detect enriched data ──
    # Check the first success criterion for enriched fields
    first_sc = None
    for p in raw_data:
        for g in p.get("guidelines", []):
            for sc in g.get("success_criteria", []):
                first_sc = sc
                break
            if first_sc:
                break
        if first_sc:
            break
    is_enriched = first_sc is not None and "techniques" in first_sc
    metrics.enriched = is_enriched
    if is_enriched:
        log.info("  Detected ENRICHED JSON — extracting techniques, examples, test rules, benefits")
    else:
        log.info("  Detected BASE JSON — enriched fields will be empty")

    # ── Static conformance levels ──
    conformance_levels = [
        {"name": "A",   "description": "Minimum level of conformance"},
        {"name": "AA",  "description": "Addresses most common barriers — target for most organizations"},
        {"name": "AAA", "description": "Highest level of conformance"},
    ]

    principles = []
    guidelines = []
    criteria = []
    special_cases = []
    notes_list = []
    references = {}          # url → {title, url}  (deduplicated)
    guideline_refs = []
    criterion_refs = []

    # ── Enriched collections ──
    techniques_map = {}      # tech_id → technique node dict (deduplicated globally)
    technique_edges = []     # {criterion_id, tech_id, relationship}
    test_rules_list = []     # {criterion_id, title, url}
    examples_list = []       # {criterion_id, index, title, description}
    benefits_list = []       # {criterion_id, index, description}
    enriched_cross_refs = [] # {source, target, rel, desc}
    key_terms_list = []      # {criterion_id, index, term, definition}
    related_resources_list = []  # {criterion_id, title, url}

    # ── Build set of valid SC ref_ids for cross-ref validation ──
    valid_ref_ids = set()
    for p in raw_data:
        for g in p.get("guidelines", []):
            for sc in g.get("success_criteria", []):
                valid_ref_ids.add(sc["ref_id"])

    for principle in raw_data:
        principles.append({
            "ref_id": principle["ref_id"],
            "title": principle["title"],
            "description": principle["description"],
            "url": principle["url"],
        })
        metrics.principles += 1

        for guideline in principle.get("guidelines", []):
            guidelines.append({
                "ref_id": guideline["ref_id"],
                "title": guideline["title"],
                "description": guideline["description"],
                "url": guideline["url"],
                "principle_id": principle["ref_id"],
            })
            metrics.guidelines += 1

            # Guideline-level references
            for ref in guideline.get("references", []):
                references[ref["url"]] = {"title": ref["title"], "url": ref["url"]}
                guideline_refs.append({
                    "guideline_id": guideline["ref_id"],
                    "url": ref["url"],
                    "title": ref["title"],
                })
                metrics.references += 1

            for sc in guideline.get("success_criteria", []):
                # ── Base criterion record (always present) ──
                criterion_record = {
                    "ref_id": sc["ref_id"],
                    "title": sc["title"],
                    "description": sc["description"],
                    "url": sc["url"],
                    "level": sc["level"],
                    "guideline_id": guideline["ref_id"],
                    "principle_id": principle["ref_id"],
                    "principle_title": principle["title"],
                    "guideline_title": guideline["title"],
                    "full_label": f"{sc['ref_id']} {sc['title']} (Level {sc['level']})",
                }

                # ── Enriched criterion properties (if present) ──
                if is_enriched:
                    in_brief = sc.get("in_brief", {})
                    criterion_record.update({
                        "intent": sc.get("intent") or "",
                        "wcag_version": sc.get("wcag_version", "2.0"),
                        "automatable": sc.get("automatable", "manual"),
                        "disability_impact": sc.get("disability_impact", []),
                        "input_types_affected": sc.get("input_types_affected", []),
                        "technology_applicability": sc.get("technology_applicability", []),
                        "in_brief_goal": in_brief.get("goal", ""),
                        "in_brief_what_to_do": in_brief.get("what_to_do", ""),
                        "in_brief_why_important": in_brief.get("why_important", ""),
                    })

                criteria.append(criterion_record)
                metrics.criteria += 1

                # Special cases
                for idx, case in enumerate(sc.get("special_cases") or []):
                    special_cases.append({
                        "criterion_id": sc["ref_id"],
                        "index": idx,
                        "type": case.get("type", "unknown"),
                        "title": case.get("title", ""),
                        "description": case.get("description", ""),
                    })
                    metrics.special_cases += 1

                # Notes
                for idx, note in enumerate(sc.get("notes") or []):
                    notes_list.append({
                        "criterion_id": sc["ref_id"],
                        "index": idx,
                        "content": note.get("content", ""),
                    })
                    metrics.notes += 1

                # Criterion-level references
                for ref in sc.get("references", []):
                    references[ref["url"]] = {"title": ref["title"], "url": ref["url"]}
                    criterion_refs.append({
                        "criterion_id": sc["ref_id"],
                        "url": ref["url"],
                        "title": ref["title"],
                    })
                    metrics.references += 1

                # ── Enriched: Techniques ──
                if is_enriched:
                    tech_data = sc.get("techniques", {})
                    for category, rel_type in [
                        ("sufficient", "HAS_TECHNIQUE"),
                        ("advisory", "HAS_ADVISORY_TECHNIQUE"),
                        ("failures", "HAS_FAILURE"),
                    ]:
                        for tech in tech_data.get(category, []):
                            tech_id = tech.get("id", "")
                            if not tech_id:
                                continue
                            # Build global technique node (deduplicated)
                            if tech_id not in techniques_map:
                                techniques_map[tech_id] = {
                                    "tech_id": tech_id,
                                    "title": tech.get("title", ""),
                                    "url": tech.get("url", ""),
                                    "technology": tech.get("technology", "general"),
                                    "category": category,
                                }
                                metrics.techniques += 1
                            # Edge: criterion → technique
                            technique_edges.append({
                                "criterion_id": sc["ref_id"],
                                "tech_id": tech_id,
                                "relationship": rel_type,
                            })

                    # ── Enriched: Test Rules ──
                    for rule in sc.get("test_rules", []):
                        test_rules_list.append({
                            "criterion_id": sc["ref_id"],
                            "title": rule.get("title", ""),
                            "url": rule.get("url", ""),
                        })
                        metrics.test_rules += 1

                    # ── Enriched: Examples ──
                    for idx, ex in enumerate(sc.get("examples", [])):
                        examples_list.append({
                            "criterion_id": sc["ref_id"],
                            "index": idx,
                            "title": ex.get("title", ""),
                            "description": ex.get("description", ""),
                        })
                        metrics.examples += 1

                    # ── Enriched: Benefits ──
                    for idx, benefit in enumerate(sc.get("benefits", [])):
                        benefit_text = benefit if isinstance(benefit, str) else str(benefit)
                        benefits_list.append({
                            "criterion_id": sc["ref_id"],
                            "index": idx,
                            "description": benefit_text,
                        })
                        metrics.benefits += 1

                    # ── Enriched: Dynamic cross-references (from Understanding pages) ──
                    for related_id in sc.get("related_scs", []):
                        # Filter out invalid ref_ids (e.g. 5.2.1, 4.8.5 don't exist)
                        if related_id in valid_ref_ids:
                            enriched_cross_refs.append({
                                "source": sc["ref_id"],
                                "target": related_id,
                                "rel": "RELATED_CRITERION",
                                "desc": f"Related criterion identified from Understanding {sc['ref_id']}",
                            })

                    # ── Enriched: Key Terms ──
                    for idx, kt in enumerate(sc.get("key_terms", [])):
                        key_terms_list.append({
                            "criterion_id": sc["ref_id"],
                            "index": idx,
                            "term": kt.get("term", ""),
                            "definition": kt.get("definition", ""),
                        })
                        metrics.key_terms += 1

                    # ── Enriched: Related Resources ──
                    for rr in sc.get("related_resources", []):
                        related_resources_list.append({
                            "criterion_id": sc["ref_id"],
                            "title": rr.get("title", ""),
                            "url": rr.get("url", ""),
                        })
                        metrics.related_resources += 1

    # ── Static cross-references (always present) ──
    cross_refs = [
        {"source": "1.3.3", "target": "1.4",   "rel": "RELATED_GUIDELINE", "desc": "For color requirements, refer to Guideline 1.4"},
        {"source": "1.4.1", "target": "1.3",   "rel": "RELATED_GUIDELINE", "desc": "Other perception forms covered in Guideline 1.3"},
        {"source": "2.2.1", "target": "3.2.1", "rel": "RELATED_CRITERION", "desc": "Consider in conjunction with 3.2.1"},
        {"source": "2.2.2", "target": "2.3",   "rel": "RELATED_GUIDELINE", "desc": "For flickering content, refer to Guideline 2.3"},
        {"source": "2.4.10","target": "4.1.2", "rel": "RELATED_CRITERION", "desc": "UI components covered under 4.1.2"},
        {"source": "4.1.1", "target": "4.1.2", "rel": "RELATED_CRITERION", "desc": "Related parsing and name/role/value requirements"},
        {"source": "1.1.1", "target": "4.1",   "rel": "RELATED_GUIDELINE", "desc": "Additional requirements for controls in Guideline 4.1"},
        {"source": "1.1.1", "target": "1.2",   "rel": "RELATED_GUIDELINE", "desc": "Additional requirements for media in Guideline 1.2"},
    ]
    metrics.cross_refs = len(cross_refs)

    log.info(
        "  Transformed → %d principles, %d guidelines, %d criteria, "
        "%d special cases, %d notes, %d references",
        metrics.principles, metrics.guidelines, metrics.criteria,
        metrics.special_cases, metrics.notes, metrics.references,
    )
    if is_enriched:
        log.info(
            "  Enriched    → %d techniques, %d test rules, %d examples, "
            "%d benefits, %d key terms, %d related resources, %d dynamic cross-refs",
            metrics.techniques, metrics.test_rules, metrics.examples,
            metrics.benefits, metrics.key_terms, metrics.related_resources,
            len(enriched_cross_refs),
        )

    return TransformedData(
        conformance_levels=conformance_levels,
        principles=principles,
        guidelines=guidelines,
        criteria=criteria,
        special_cases=special_cases,
        notes=notes_list,
        references=list(references.values()),
        guideline_refs=guideline_refs,
        criterion_refs=criterion_refs,
        cross_refs=cross_refs,
        techniques=list(techniques_map.values()),
        technique_edges=technique_edges,
        test_rules=test_rules_list,
        examples=examples_list,
        benefits=benefits_list,
        enriched_cross_refs=enriched_cross_refs,
        key_terms=key_terms_list,
        related_resources=related_resources_list,
    )


# ================================================================
#  PHASE 3 — LOAD: Batch-write into Neo4j
# ================================================================
def phase_load(db: Neo4jConnection, data: TransformedData):
    """Load all transformed records into Neo4j using batched writes."""
    log.info("PHASE 3 ▸ Load — writing to Neo4j")

    # ── 3a. Conformance levels ──
    db.batch_write("""
        UNWIND $rows AS row
        MERGE (cl:ConformanceLevel {name: row.name})
        SET cl.description = row.description
    """, data.conformance_levels)
    log.info("  Loaded %d conformance levels", len(data.conformance_levels))

    # ── 3b. Principles ──
    db.batch_write("""
        UNWIND $rows AS row
        MERGE (p:WCAGPrinciple {ref_id: row.ref_id})
        SET p.name        = row.title,
            p.title       = row.title,
            p.description = row.description,
            p.url         = row.url
    """, data.principles)
    log.info("  Loaded %d principles", len(data.principles))

    # ── 3c. Guidelines + PART_OF → Principle ──
    db.batch_write("""
        UNWIND $rows AS row
        MERGE (g:WCAGGuideline {ref_id: row.ref_id})
        SET g.title        = row.title,
            g.description  = row.description,
            g.url          = row.url,
            g.principle_id = row.principle_id
        WITH g, row
        MATCH (p:WCAGPrinciple {ref_id: row.principle_id})
        MERGE (g)-[:PART_OF]->(p)
    """, data.guidelines)
    log.info("  Loaded %d guidelines", len(data.guidelines))

    # ── 3d. Criteria + PART_OF → Guideline + HAS_LEVEL → ConformanceLevel ──
    # Enriched properties are set conditionally (null-safe via COALESCE)
    db.batch_write("""
        UNWIND $rows AS row
        MERGE (c:WCAGCriterion {ref_id: row.ref_id})
        SET c.code             = row.ref_id,
            c.title            = row.title,
            c.name             = row.title,
            c.description      = row.description,
            c.url              = row.url,
            c.level            = row.level,
            c.guideline_id     = row.guideline_id,
            c.principle_id     = row.principle_id,
            c.principle_title  = row.principle_title,
            c.guideline_title  = row.guideline_title,
            c.full_label       = row.full_label,
            c.intent           = COALESCE(row.intent, ''),
            c.wcag_version     = COALESCE(row.wcag_version, '2.0'),
            c.automatable      = COALESCE(row.automatable, 'manual'),
            c.disability_impact       = COALESCE(row.disability_impact, []),
            c.input_types_affected    = COALESCE(row.input_types_affected, []),
            c.technology_applicability = COALESCE(row.technology_applicability, []),
            c.in_brief_goal           = COALESCE(row.in_brief_goal, ''),
            c.in_brief_what_to_do     = COALESCE(row.in_brief_what_to_do, ''),
            c.in_brief_why_important  = COALESCE(row.in_brief_why_important, '')
        WITH c, row
        MATCH (g:WCAGGuideline {ref_id: row.guideline_id})
        MERGE (c)-[:PART_OF]->(g)
        WITH c, row
        MATCH (cl:ConformanceLevel {name: row.level})
        MERGE (c)-[:HAS_LEVEL]->(cl)
    """, data.criteria)
    log.info("  Loaded %d criteria", len(data.criteria))

    # ── 3e. Special cases ──
    if data.special_cases:
        db.batch_write("""
            UNWIND $rows AS row
            MATCH (c:WCAGCriterion {ref_id: row.criterion_id})
            CREATE (s:WCAGSpecialCase {
                criterion_id: row.criterion_id,
                index:        row.index,
                type:         row.type,
                title:        row.title,
                description:  row.description
            })
            CREATE (c)-[:HAS_SPECIAL_CASE]->(s)
        """, data.special_cases)
    log.info("  Loaded %d special cases", len(data.special_cases))

    # ── 3f. Notes ──
    if data.notes:
        db.batch_write("""
            UNWIND $rows AS row
            MATCH (c:WCAGCriterion {ref_id: row.criterion_id})
            CREATE (n:WCAGNote {
                criterion_id: row.criterion_id,
                index:        row.index,
                content:      row.content
            })
            CREATE (c)-[:HAS_NOTE]->(n)
        """, data.notes)
    log.info("  Loaded %d notes", len(data.notes))

    # ── 3g. Reference nodes (deduplicated) ──
    if data.references:
        db.batch_write("""
            UNWIND $rows AS row
            MERGE (r:WCAGReference {url: row.url})
            SET r.title = row.title
        """, data.references)
    log.info("  Loaded %d unique reference nodes", len(data.references))

    # ── 3h. Guideline → Reference edges ──
    if data.guideline_refs:
        db.batch_write("""
            UNWIND $rows AS row
            MATCH (g:WCAGGuideline {ref_id: row.guideline_id})
            MATCH (r:WCAGReference  {url:    row.url})
            MERGE (g)-[:HAS_REFERENCE]->(r)
        """, data.guideline_refs)
    log.info("  Linked %d guideline references", len(data.guideline_refs))

    # ── 3i. Criterion → Reference edges ──
    if data.criterion_refs:
        db.batch_write("""
            UNWIND $rows AS row
            MATCH (c:WCAGCriterion {ref_id: row.criterion_id})
            MATCH (r:WCAGReference  {url:    row.url})
            MERGE (c)-[:HAS_REFERENCE]->(r)
        """, data.criterion_refs)
    log.info("  Linked %d criterion references", len(data.criterion_refs))

    # ── 3j. Cross-reference relationships (static) ──
    for rel_type in ("RELATED_CRITERION", "RELATED_GUIDELINE"):
        batch = [r for r in data.cross_refs if r["rel"] == rel_type]
        if not batch:
            continue
        for row in batch:
            try:
                db.write(f"""
                    MATCH (a:WCAGCriterion {{ref_id: $source}})
                    MATCH (b) WHERE (b:WCAGCriterion OR b:WCAGGuideline)
                          AND b.ref_id = $target
                    MERGE (a)-[r:{rel_type}]->(b)
                    SET r.description = $desc
                """, {"source": row["source"], "target": row["target"], "desc": row["desc"]})
            except Exception as e:
                log.warning("  Cross-ref %s → %s failed: %s", row["source"], row["target"], e)
    log.info("  Created %d static cross-reference relationships", len(data.cross_refs))

    # ================================================================
    #  ENRICHED DATA LOADING (only when enriched JSON is used)
    # ================================================================

    # ── 3k. Technique nodes (globally deduplicated by tech_id) ──
    if data.techniques:
        db.batch_write("""
            UNWIND $rows AS row
            MERGE (t:WCAGTechnique {tech_id: row.tech_id})
            SET t.title      = row.title,
                t.url        = row.url,
                t.technology = row.technology,
                t.category   = row.category
        """, data.techniques)
        log.info("  Loaded %d technique nodes", len(data.techniques))

    # ── 3l. Criterion → Technique edges (3 relationship types) ──
    if data.technique_edges:
        for rel_type in ("HAS_TECHNIQUE", "HAS_ADVISORY_TECHNIQUE", "HAS_FAILURE"):
            batch = [e for e in data.technique_edges if e["relationship"] == rel_type]
            if not batch:
                continue
            db.batch_write(f"""
                UNWIND $rows AS row
                MATCH (c:WCAGCriterion {{ref_id: row.criterion_id}})
                MATCH (t:WCAGTechnique {{tech_id: row.tech_id}})
                MERGE (c)-[:{rel_type}]->(t)
            """, batch)
            log.info("  Linked %d %s edges", len(batch), rel_type)

    # ── 3m. Test Rule nodes + edges ──
    if data.test_rules:
        db.batch_write("""
            UNWIND $rows AS row
            MATCH (c:WCAGCriterion {ref_id: row.criterion_id})
            CREATE (tr:WCAGTestRule {
                criterion_id: row.criterion_id,
                title:        row.title,
                url:          row.url
            })
            CREATE (c)-[:HAS_TEST_RULE]->(tr)
        """, data.test_rules)
        log.info("  Loaded %d test rules", len(data.test_rules))

    # ── 3n. Example nodes + edges ──
    if data.examples:
        db.batch_write("""
            UNWIND $rows AS row
            MATCH (c:WCAGCriterion {ref_id: row.criterion_id})
            CREATE (ex:WCAGExample {
                criterion_id: row.criterion_id,
                index:        row.index,
                title:        row.title,
                description:  row.description
            })
            CREATE (c)-[:HAS_EXAMPLE]->(ex)
        """, data.examples)
        log.info("  Loaded %d examples", len(data.examples))

    # ── 3o. Benefit nodes + edges ──
    if data.benefits:
        db.batch_write("""
            UNWIND $rows AS row
            MATCH (c:WCAGCriterion {ref_id: row.criterion_id})
            CREATE (b:WCAGBenefit {
                criterion_id: row.criterion_id,
                index:        row.index,
                description:  row.description
            })
            CREATE (c)-[:HAS_BENEFIT]->(b)
        """, data.benefits)
        log.info("  Loaded %d benefits", len(data.benefits))

    # ── 3p. Dynamic cross-references (from enriched Understanding pages) ──
    if data.enriched_cross_refs:
        for row in data.enriched_cross_refs:
            try:
                db.write("""
                    MATCH (a:WCAGCriterion {ref_id: $source})
                    MATCH (b:WCAGCriterion {ref_id: $target})
                    MERGE (a)-[r:RELATED_CRITERION]->(b)
                    SET r.description = $desc, r.source = 'understanding_page'
                """, {"source": row["source"], "target": row["target"], "desc": row["desc"]})
            except Exception as e:
                log.warning("  Dynamic cross-ref %s → %s failed: %s", row["source"], row["target"], e)
        log.info("  Created %d dynamic cross-reference relationships", len(data.enriched_cross_refs))

    # ── 3q. Key Term nodes + edges ──
    if data.key_terms:
        db.batch_write("""
            UNWIND $rows AS row
            MATCH (c:WCAGCriterion {ref_id: row.criterion_id})
            CREATE (kt:WCAGKeyTerm {
                criterion_id: row.criterion_id,
                index:        row.index,
                term:         row.term,
                definition:   row.definition
            })
            CREATE (c)-[:HAS_KEY_TERM]->(kt)
        """, data.key_terms)
        log.info("  Loaded %d key terms", len(data.key_terms))

    # ── 3r. Related Resource nodes + edges ──
    if data.related_resources:
        db.batch_write("""
            UNWIND $rows AS row
            MATCH (c:WCAGCriterion {ref_id: row.criterion_id})
            CREATE (rr:WCAGRelatedResource {
                criterion_id: row.criterion_id,
                title:        row.title,
                url:          row.url
            })
            CREATE (c)-[:HAS_RELATED_RESOURCE]->(rr)
        """, data.related_resources)
        log.info("  Loaded %d related resources", len(data.related_resources))


# ================================================================
#  PHASE 4 — VALIDATE: Post-load integrity checks
# ================================================================
def phase_validate(db: Neo4jConnection, metrics: PipelineMetrics) -> bool:
    """Run post-load integrity checks against Neo4j. Returns True if all pass."""
    log.info("PHASE 4 ▸ Validate — integrity checks")
    passed = True

    # ── 4a. Node counts ──
    node_stats = db.read("""
        MATCH (n)
        WHERE n:WCAGPrinciple OR n:WCAGGuideline OR n:WCAGCriterion
           OR n:WCAGSpecialCase OR n:WCAGNote OR n:WCAGReference
           OR n:ConformanceLevel OR n:WCAGTechnique OR n:WCAGTestRule
           OR n:WCAGExample OR n:WCAGBenefit OR n:WCAGKeyTerm
           OR n:WCAGRelatedResource
        RETURN labels(n)[0] AS label, count(n) AS count
        ORDER BY count DESC
    """)
    log.info("  Neo4j node counts:")
    for row in node_stats:
        log.info("    %-25s %d", row["label"], row["count"])

    # ── 4b. Complete Criterion → Guideline → Principle chains ──
    broken_chain = db.read("""
        MATCH (c:WCAGCriterion)
        WHERE NOT (c)-[:PART_OF]->(:WCAGGuideline)-[:PART_OF]->(:WCAGPrinciple)
        RETURN c.ref_id AS ref_id, c.title AS title
    """)
    if broken_chain:
        passed = False
        log.error("  FAIL — %d criteria with broken hierarchy chain:", len(broken_chain))
        for row in broken_chain:
            log.error("    %s: %s", row["ref_id"], row["title"])
    else:
        log.info("  PASS — All %d criteria have complete Criterion → Guideline → Principle chains",
                 metrics.criteria)

    # ── 4c. code == ref_id on all criteria ──
    mismatched = db.read("""
        MATCH (c:WCAGCriterion)
        WHERE c.code <> c.ref_id OR c.code IS NULL
        RETURN c.ref_id AS ref_id, c.code AS code
    """)
    if mismatched:
        passed = False
        log.error("  FAIL — %d criteria have mismatched code/ref_id", len(mismatched))
    else:
        log.info("  PASS — All criteria have code == ref_id (no duplicate risk)")

    # ── 4d. Every criterion has a conformance level ──
    level_stats = db.read("""
        MATCH (c:WCAGCriterion)-[:HAS_LEVEL]->(cl:ConformanceLevel)
        WITH cl.name AS level, count(c) AS count
        RETURN level, count
        ORDER BY level
    """)
    total_leveled = sum(r["count"] for r in level_stats)
    if total_leveled != metrics.criteria:
        passed = False
        log.error("  FAIL — %d/%d criteria linked to a conformance level",
                  total_leveled, metrics.criteria)
    else:
        log.info("  PASS — Conformance level distribution:")
        for row in level_stats:
            log.info("    Level %-5s %d criteria", row["level"], row["count"])

    # ── 4e. Enriched data validation (only when enriched JSON was used) ──
    if metrics.enriched:
        log.info("  Enriched data validation:")

        # Technique nodes loaded
        tech_count = db.read("MATCH (t:WCAGTechnique) RETURN count(t) AS count")[0]["count"]
        if tech_count == 0:
            passed = False
            log.error("  FAIL — No WCAGTechnique nodes found (expected %d)", metrics.techniques)
        else:
            log.info("    PASS — %d WCAGTechnique nodes loaded", tech_count)

        # Every technique should be linked to at least one criterion
        orphan_techniques = db.read("""
            MATCH (t:WCAGTechnique)
            WHERE NOT ()-[:HAS_TECHNIQUE|HAS_ADVISORY_TECHNIQUE|HAS_FAILURE]->(t)
            RETURN count(t) AS count
        """)[0]["count"]
        if orphan_techniques > 0:
            log.warning("  WARN — %d orphan WCAGTechnique nodes (no criterion link)", orphan_techniques)

        # Criteria with enriched properties
        enriched_criteria = db.read("""
            MATCH (c:WCAGCriterion)
            WHERE c.intent IS NOT NULL AND c.intent <> ''
            RETURN count(c) AS count
        """)[0]["count"]
        log.info("    %d/%d criteria have intent text", enriched_criteria, metrics.criteria)

        # Criteria with wcag_version set
        versioned = db.read("""
            MATCH (c:WCAGCriterion)
            WHERE c.wcag_version IS NOT NULL
            WITH c.wcag_version AS version, count(c) AS count
            RETURN version, count ORDER BY version
        """)
        if versioned:
            log.info("    WCAG version distribution:")
            for row in versioned:
                log.info("      WCAG %-5s %d criteria", row["version"], row["count"])

        # Automation classification
        automatable_stats = db.read("""
            MATCH (c:WCAGCriterion)
            WHERE c.automatable IS NOT NULL
            WITH c.automatable AS level, count(c) AS count
            RETURN level, count ORDER BY level
        """)
        if automatable_stats:
            log.info("    Automation classification:")
            for row in automatable_stats:
                log.info("      %-10s %d criteria", row["level"], row["count"])

        # Test rules, examples, benefits, key terms, related resources counts
        for label in ["WCAGTestRule", "WCAGExample", "WCAGBenefit",
                       "WCAGKeyTerm", "WCAGRelatedResource"]:
            count = db.read(f"MATCH (n:{label}) RETURN count(n) AS count")[0]["count"]
            log.info("    %d %s nodes", count, label)

    return passed


# ================================================================
#  PIPELINE ORCHESTRATOR
# ================================================================
def run_pipeline():
    """
    Execute the full ETL pipeline with phase tracking, timing, and
    structured error handling.
    """
    log.info("=" * 62)
    log.info("  WCAG 2.2 Foundation — ETL Pipeline")
    log.info("=" * 62)

    pipeline_start = time.time()
    config = PipelineConfig()
    metrics = PipelineMetrics()
    db = None

    try:
        # ── Config validation ──
        config.validate()

        # ── Connect ──
        db = Neo4jConnection(config)

        # ── Phase 0: Pre-flight ──
        t0 = time.time()
        phase_preflight(db)
        metrics.phase_times["0_preflight"] = round(time.time() - t0, 2)

        # ── Phase 1: Extract ──
        t0 = time.time()
        raw_data = phase_extract(config)
        metrics.phase_times["1_extract"] = round(time.time() - t0, 2)

        # ── Phase 2: Transform ──
        t0 = time.time()
        transformed = phase_transform(raw_data, metrics)
        metrics.phase_times["2_transform"] = round(time.time() - t0, 2)

        # ── Phase 3: Load ──
        t0 = time.time()
        phase_load(db, transformed)
        metrics.phase_times["3_load"] = round(time.time() - t0, 2)

        # ── Phase 4: Validate ──
        t0 = time.time()
        all_passed = phase_validate(db, metrics)
        metrics.phase_times["4_validate"] = round(time.time() - t0, 2)

        # ── Summary ──
        total_time = round(time.time() - pipeline_start, 2)
        log.info("")
        log.info("=" * 62)
        log.info("  PIPELINE COMPLETE — %.2fs total", total_time)
        log.info("=" * 62)
        log.info("  Phase timing:")
        for phase, elapsed in metrics.phase_times.items():
            log.info("    %-20s %.2fs", phase, elapsed)
        log.info("")
        log.info("  Record counts:")
        for key, val in metrics.summary().items():
            log.info("    %-20s %s", key, val)
        log.info("")

        if all_passed:
            log.info("✅ All validation checks passed")
            log.info("   Next: python 02_pipeline_etl_jira.py")
        else:
            log.warning("⚠️  Some validation checks failed — review logs above")
            sys.exit(1)

    except (EnvironmentError, ConnectionError) as e:
        log.error("PIPELINE ABORTED (config/connection): %s", e)
        sys.exit(1)
    except FileNotFoundError as e:
        log.error("PIPELINE ABORTED (extract): %s", e)
        sys.exit(1)
    except ValueError as e:
        log.error("PIPELINE ABORTED (validation): %s", e)
        sys.exit(1)
    except Exception as e:
        log.error("PIPELINE ABORTED (unexpected): %s", e, exc_info=True)
        sys.exit(1)
    finally:
        if db:
            db.close()
            log.info("Neo4j connection closed")


# ================================================================
#  ENTRY POINT
# ================================================================
if __name__ == "__main__":
    run_pipeline()