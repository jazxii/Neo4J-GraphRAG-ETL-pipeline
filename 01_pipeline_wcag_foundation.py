"""
Pipeline Step 1: Load WCAG 2.2 Specification into Neo4j.

This runs FIRST — before any JIRA data.
Creates the authoritative WCAG knowledge layer that all bugs
will link to via :VIOLATES relationships.

Node Types:
  - WCAGPrinciple (4): Perceivable, Operable, Understandable, Robust
  - WCAGGuideline (13): 1.1 through 4.1
  - WCAGCriterion (86+): 1.1.1 through 4.1.3
  - ConformanceLevel (3): A, AA, AAA
  - WCAGSpecialCase: Exceptions/conditions per criterion
  - WCAGNote: Clarifying notes per criterion
  - WCAGReference: How to Meet / Understanding links

Usage:
  python pipeline_step1_wcag_foundation.py
"""

from neo4j import GraphDatabase
import json
import sys
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# ============================================================
# CONFIGURATION
# ============================================================
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

WCAG_JSON_FILE = os.getenv("WCAG_JSON_FILE", "wcag_22_guidelines.json")

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))


def run_query(query, params=None):
    with driver.session() as session:
        return [r.data() for r in session.run(query, params or {})]


def run_write(query, params=None):
    """Execute a write query without consuming results."""
    with driver.session() as session:
        session.run(query, params or {})


# ============================================================
# STEP 1: CLEAN SLATE FOR WCAG NODES (optional, safe)
# ============================================================
print("🧹 Cleaning existing WCAG nodes (if any)...")

# Remove old WCAG data to avoid duplicates on re-run
for label in ['WCAGSpecialCase', 'WCAGNote', 'WCAGReference']:
    run_write(f"MATCH (n:{label}) DETACH DELETE n")

# Don't delete WCAGCriterion/Guideline/Principle yet — bugs may link to them
# Instead we'll MERGE (upsert) below

print("   ✅ Cleaned auxiliary WCAG nodes")


# ============================================================
# STEP 2: CREATE CONSTRAINTS & INDEXES
# Use ref_id as the SINGLE unique key for WCAGCriterion.
# Also store 'code' as a synonym (same value), but ref_id is primary.
# ============================================================
print("\n📐 Creating constraints and indexes...")

constraints = [
    "CREATE CONSTRAINT IF NOT EXISTS FOR (p:WCAGPrinciple) REQUIRE p.ref_id IS UNIQUE",
    "CREATE CONSTRAINT IF NOT EXISTS FOR (g:WCAGGuideline) REQUIRE g.ref_id IS UNIQUE",
    "CREATE CONSTRAINT IF NOT EXISTS FOR (c:WCAGCriterion) REQUIRE c.ref_id IS UNIQUE",
    "CREATE CONSTRAINT IF NOT EXISTS FOR (cl:ConformanceLevel) REQUIRE cl.name IS UNIQUE",
    "CREATE INDEX IF NOT EXISTS FOR (c:WCAGCriterion) ON (c.code)",
    "CREATE INDEX IF NOT EXISTS FOR (c:WCAGCriterion) ON (c.title)",
    "CREATE INDEX IF NOT EXISTS FOR (c:WCAGCriterion) ON (c.level)",
    "CREATE INDEX IF NOT EXISTS FOR (g:WCAGGuideline) ON (g.title)",
    "CREATE INDEX IF NOT EXISTS FOR (p:WCAGPrinciple) ON (p.name)",
]

for c in constraints:
    try:
        run_write(c)
    except Exception as e:
        # Constraint may already exist, that's fine
        if "already exists" not in str(e).lower():
            print(f"   ⚠️  {e}")

print("   ✅ Constraints and indexes ready")


# ============================================================
# STEP 3: CONFORMANCE LEVELS
# ============================================================
print("\n🏷️  Creating conformance level nodes...")

for level, desc in [
    ("A", "Minimum level of conformance"),
    ("AA", "Addresses most common barriers — target for most organizations"),
    ("AAA", "Highest level of conformance"),
]:
    run_write("""
        MERGE (cl:ConformanceLevel {name: $level})
        SET cl.description = $desc
    """, {"level": level, "desc": desc})

print("   ✅ Conformance levels: A, AA, AAA")


# ============================================================
# STEP 4: LOAD WCAG JSON
# ============================================================
print(f"\n📂 Loading {WCAG_JSON_FILE}...")

try:
    with open(WCAG_JSON_FILE, "r", encoding="utf-8") as f:
        wcag_data = json.load(f)
except FileNotFoundError:
    print(f"   ❌ File not found: {WCAG_JSON_FILE}")
    print(f"      Place your WCAG 2.2 JSON file in the same directory.")
    sys.exit(1)

print(f"   ✅ Loaded {len(wcag_data)} principles")


# ============================================================
# STEP 5: CREATE PRINCIPLES → GUIDELINES → CRITERIA
# ============================================================
counts = {
    "principles": 0, "guidelines": 0, "criteria": 0,
    "special_cases": 0, "notes": 0, "references": 0,
}

for principle in wcag_data:
    # ── Principle ──
    run_write("""
        MERGE (p:WCAGPrinciple {ref_id: $ref_id})
        SET p.name = $title,
            p.title = $title,
            p.description = $description,
            p.url = $url
    """, {
        "ref_id": principle["ref_id"],
        "title": principle["title"],
        "description": principle["description"],
        "url": principle["url"],
    })
    counts["principles"] += 1
    print(f"\n   📌 Principle {principle['ref_id']}: {principle['title']}")

    for guideline in principle.get("guidelines", []):
        # ── Guideline ──
        run_write("""
            MERGE (g:WCAGGuideline {ref_id: $ref_id})
            SET g.title = $title,
                g.description = $description,
                g.url = $url,
                g.principle_id = $principle_id

            WITH g
            MATCH (p:WCAGPrinciple {ref_id: $principle_id})
            MERGE (g)-[:PART_OF]->(p)
        """, {
            "ref_id": guideline["ref_id"],
            "title": guideline["title"],
            "description": guideline["description"],
            "url": guideline["url"],
            "principle_id": principle["ref_id"],
        })
        counts["guidelines"] += 1
        print(f"      📋 Guideline {guideline['ref_id']}: {guideline['title']}")

        # Guideline references
        for ref in guideline.get("references", []):
            run_write("""
                MATCH (g:WCAGGuideline {ref_id: $gid})
                MERGE (r:WCAGReference {url: $url})
                SET r.title = $title
                MERGE (g)-[:HAS_REFERENCE]->(r)
            """, {"gid": guideline["ref_id"], "title": ref["title"], "url": ref["url"]})
            counts["references"] += 1

        for sc in guideline.get("success_criteria", []):
            # ── Criterion ──
            # KEY: We set BOTH ref_id AND code to the same value.
            # This prevents the v1 duplicate problem entirely.
            run_write("""
                MERGE (c:WCAGCriterion {ref_id: $ref_id})
                SET c.code = $ref_id,
                    c.title = $title,
                    c.name = $title,
                    c.description = $description,
                    c.url = $url,
                    c.level = $level,
                    c.guideline_id = $guideline_id,
                    c.principle_id = $principle_id,
                    c.principle_title = $principle_title,
                    c.guideline_title = $guideline_title,
                    c.full_label = $ref_id + ' ' + $title + ' (Level ' + $level + ')'

                WITH c
                MATCH (g:WCAGGuideline {ref_id: $guideline_id})
                MERGE (c)-[:PART_OF]->(g)

                WITH c
                MATCH (cl:ConformanceLevel {name: $level})
                MERGE (c)-[:HAS_LEVEL]->(cl)
            """, {
                "ref_id": sc["ref_id"],
                "title": sc["title"],
                "description": sc["description"],
                "url": sc["url"],
                "level": sc["level"],
                "guideline_id": guideline["ref_id"],
                "principle_id": principle["ref_id"],
                "principle_title": principle["title"],
                "guideline_title": guideline["title"],
            })
            counts["criteria"] += 1

            # Special cases
            for i, case in enumerate(sc.get("special_cases") or []):
                run_write("""
                    MATCH (c:WCAGCriterion {ref_id: $cid})
                    CREATE (s:WCAGSpecialCase {
                        criterion_id: $cid, index: $idx,
                        type: $type, title: $title, description: $desc
                    })
                    CREATE (c)-[:HAS_SPECIAL_CASE]->(s)
                """, {
                    "cid": sc["ref_id"], "idx": i,
                    "type": case.get("type", "unknown"),
                    "title": case.get("title", ""),
                    "desc": case.get("description", ""),
                })
                counts["special_cases"] += 1

            # Notes
            for i, note in enumerate(sc.get("notes") or []):
                run_write("""
                    MATCH (c:WCAGCriterion {ref_id: $cid})
                    CREATE (n:WCAGNote {
                        criterion_id: $cid, index: $idx,
                        content: $content
                    })
                    CREATE (c)-[:HAS_NOTE]->(n)
                """, {
                    "cid": sc["ref_id"], "idx": i,
                    "content": note.get("content", ""),
                })
                counts["notes"] += 1

            # References
            for ref in sc.get("references", []):
                run_write("""
                    MATCH (c:WCAGCriterion {ref_id: $cid})
                    MERGE (r:WCAGReference {url: $url})
                    SET r.title = $title
                    MERGE (c)-[:HAS_REFERENCE]->(r)
                """, {"cid": sc["ref_id"], "title": ref["title"], "url": ref["url"]})
                counts["references"] += 1

            print(f"         ✅ {sc['ref_id']} {sc['title']} (Level {sc['level']})")


# ============================================================
# STEP 6: CROSS-REFERENCES BETWEEN CRITERIA
# ============================================================
print("\n🔗 Creating cross-reference relationships...")

CROSS_REFS = [
    ("1.3.3", "1.4", "RELATED_GUIDELINE", "For color requirements, refer to Guideline 1.4"),
    ("1.4.1", "1.3", "RELATED_GUIDELINE", "Other perception forms covered in Guideline 1.3"),
    ("2.2.1", "3.2.1", "RELATED_CRITERION", "Consider in conjunction with 3.2.1"),
    ("2.2.2", "2.3", "RELATED_GUIDELINE", "For flickering content, refer to Guideline 2.3"),
    ("2.4.10", "4.1.2", "RELATED_CRITERION", "UI components covered under 4.1.2"),
    ("4.1.1", "4.1.2", "RELATED_CRITERION", "Related parsing and name/role/value requirements"),
    ("1.1.1", "4.1", "RELATED_GUIDELINE", "Additional requirements for controls in Guideline 4.1"),
    ("1.1.1", "1.2", "RELATED_GUIDELINE", "Additional requirements for media in Guideline 1.2"),
]

for source, target, rel_type, desc in CROSS_REFS:
    try:
        run_write(f"""
            MATCH (a:WCAGCriterion {{ref_id: $source}})
            MATCH (b) WHERE (b:WCAGCriterion OR b:WCAGGuideline) AND b.ref_id = $target
            MERGE (a)-[r:{rel_type}]->(b)
            SET r.description = $desc
        """, {"source": source, "target": target, "desc": desc})
        print(f"   ✅ {source} ──{rel_type}──▶ {target}")
    except Exception as e:
        print(f"   ⚠️  {source} → {target}: {e}")


# ============================================================
# STEP 7: VERIFICATION
# ============================================================
print("\n" + "=" * 60)
print("📊 WCAG FOUNDATION — VERIFICATION")
print("=" * 60)

print(f"\n   Loaded from JSON:")
for k, v in counts.items():
    print(f"   {k:<20} {v}")

# Verify in Neo4j
node_stats = run_query("""
    MATCH (n)
    WHERE n:WCAGPrinciple OR n:WCAGGuideline OR n:WCAGCriterion
       OR n:WCAGSpecialCase OR n:WCAGNote OR n:WCAGReference
       OR n:ConformanceLevel
    RETURN labels(n)[0] AS label, count(n) AS count
    ORDER BY count DESC
""")
print(f"\n   Neo4j Node Counts:")
for row in node_stats:
    print(f"   {row['label']:<25} {row['count']}")

# Verify all criteria have PART_OF → Guideline → Principle chain
broken_chain = run_query("""
    MATCH (c:WCAGCriterion)
    WHERE NOT (c)-[:PART_OF]->(:WCAGGuideline)-[:PART_OF]->(:WCAGPrinciple)
    RETURN c.ref_id AS ref_id, c.title AS title
""")
if broken_chain:
    print(f"\n   ⚠️  Criteria with broken guideline/principle chain:")
    for row in broken_chain:
        print(f"      {row['ref_id']}: {row['title']}")
else:
    print(f"\n   ✅ All {counts['criteria']} criteria have complete Criterion → Guideline → Principle chains")

# Verify code == ref_id on all criteria
mismatched = run_query("""
    MATCH (c:WCAGCriterion)
    WHERE c.code <> c.ref_id OR c.code IS NULL
    RETURN c.ref_id AS ref_id, c.code AS code
""")
if mismatched:
    print(f"   ⚠️  {len(mismatched)} criteria have mismatched code/ref_id")
else:
    print(f"   ✅ All criteria have code == ref_id (no future duplicate risk)")

level_stats = run_query("""
    MATCH (c:WCAGCriterion)-[:HAS_LEVEL]->(cl:ConformanceLevel)
    WITH cl.name AS level, count(c) AS count
    RETURN level, count
    ORDER BY level
""")
print(f"\n   By Conformance Level:")
for row in level_stats:
    print(f"   Level {row['level']:<5} {row['count']} criteria")

driver.close()
print("\n✅ WCAG Foundation loaded successfully!")
print("   Next: python pipeline_step2_etl_jira.py")