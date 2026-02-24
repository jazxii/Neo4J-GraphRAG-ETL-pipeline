"""
WCAG 2.2 JSON Enrichment Script

Reads the base wcag_22_guidelines.json and enriches each success criterion
by scraping the official W3C "Understanding" pages to extract:

  - intent          (the "why" behind the criterion)
  - benefits        (who it helps and how)
  - examples        (concrete illustrations)
  - techniques.sufficient   (known ways to pass)
  - techniques.advisory     (recommended but not required)
  - techniques.failures     (known ways to fail)
  - test_rules      (ACT Rules for automated testing)
  - in_brief        (goal / what_to_do / why_important)
  - disability_impact       (which disabilities are affected)
  - input_types_affected    (keyboard / pointer / visual / auditory etc.)
  - technology_applicability (HTML / CSS / PDF / ARIA / etc.)
  - wcag_version_added      (2.0 / 2.1 / 2.2)
  - automatable     (full / partial / manual)
  - related_scs     (cross-referenced success criteria)

Usage:
  pip install requests beautifulsoup4
  python 00_enrich_wcag_json.py

Output:
  wcag_22_guidelines_enriched.json
"""

import json
import re
import sys
import time
import logging
from urllib.parse import urljoin

try:
    import requests
    from bs4 import BeautifulSoup
except ImportError:
    print("Install dependencies:  pip install requests beautifulsoup4")
    sys.exit(1)

# ── Logging ──
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-7s │ %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("wcag_enrich")

INPUT_FILE = "wcag_22_guidelines.json"
OUTPUT_FILE = "wcag_22_guidelines_enriched.json"
RATE_LIMIT_SECONDS = 1.5  # polite delay between requests

# ── SC ref_ids introduced in each WCAG version ──
WCAG_21_NEW = {
    "1.3.4", "1.3.5", "1.3.6", "1.4.10", "1.4.11", "1.4.12", "1.4.13",
    "2.1.4", "2.5.1", "2.5.2", "2.5.3", "2.5.4", "2.5.5", "2.5.6",
    "4.1.3",
}
WCAG_22_NEW = {
    "2.4.11", "2.4.12", "2.4.13",
    "2.5.7", "2.5.8",
    "3.2.6", "3.3.7", "3.3.8", "3.3.9",
}

# ── Disability mapping by principle / guideline context ──
# Comprehensive mapping based on W3C "Benefits" sections
DISABILITY_IMPACT_MAP = {
    "1.1": ["blindness", "low_vision", "deafblindness", "cognitive", "learning"],
    "1.2": ["deafness", "hard_of_hearing", "blindness", "deafblindness", "cognitive", "learning"],
    "1.3": ["blindness", "low_vision", "cognitive", "learning", "mobility"],
    "1.4": ["low_vision", "color_blindness", "cognitive", "hard_of_hearing"],
    "2.1": ["blindness", "mobility", "motor_impairment"],
    "2.2": ["blindness", "low_vision", "cognitive", "learning", "motor_impairment", "deafness"],
    "2.3": ["photosensitive_epilepsy", "vestibular_disorders"],
    "2.4": ["blindness", "low_vision", "cognitive", "learning", "motor_impairment"],
    "2.5": ["mobility", "motor_impairment", "tremor", "cognitive"],
    "3.1": ["cognitive", "learning", "deafness", "blindness"],
    "3.2": ["blindness", "cognitive", "learning", "low_vision"],
    "3.3": ["blindness", "cognitive", "learning", "motor_impairment", "low_vision"],
    "4.1": ["blindness", "low_vision", "cognitive", "motor_impairment"],
}

# ── Automatable classification ──
FULLY_AUTOMATABLE = {
    "1.3.1", "1.3.2", "1.4.3", "1.4.6", "1.4.10", "1.4.12",
    "2.4.1", "2.4.2", "3.1.1", "3.1.2", "4.1.1", "4.1.2",
}
PARTIALLY_AUTOMATABLE = {
    "1.1.1", "1.2.1", "1.2.2", "1.2.3", "1.2.4", "1.2.5",
    "1.3.3", "1.3.4", "1.3.5", "1.4.1", "1.4.4", "1.4.5",
    "1.4.11", "1.4.13",
    "2.1.1", "2.1.2", "2.1.4", "2.2.1", "2.2.2",
    "2.4.3", "2.4.4", "2.4.6", "2.4.7", "2.4.11",
    "2.5.1", "2.5.2", "2.5.3", "2.5.7", "2.5.8",
    "3.2.1", "3.2.2", "3.2.6", "3.3.1", "3.3.2", "3.3.7", "3.3.8",
    "4.1.3",
}
# Everything else → "manual"


def get_understanding_url(sc_url: str) -> str:
    """Derive the Understanding page URL from the criterion's spec URL."""
    slug = sc_url.split("#")[-1] if "#" in sc_url else ""
    return f"https://www.w3.org/WAI/WCAG22/Understanding/{slug}.html"


def fetch_page(url: str) -> BeautifulSoup | None:
    """Fetch a URL and return parsed BeautifulSoup, or None on failure."""
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        return BeautifulSoup(resp.text, "html.parser")
    except Exception as e:
        log.warning("  Failed to fetch %s: %s", url, e)
        return None


def extract_section_text(soup: BeautifulSoup, heading_id: str) -> str:
    """Extract all text under a heading (by id) until the next heading."""
    heading = soup.find(id=heading_id)
    if not heading:
        # Try finding by text content
        for h in soup.find_all(re.compile(r"^h[2-4]$")):
            if heading_id.replace("-", " ") in h.get_text().lower():
                heading = h
                break
    if not heading:
        return ""

    parts = []
    for sib in heading.find_next_siblings():
        if sib.name and re.match(r"^h[2-3]$", sib.name):
            break
        text = sib.get_text(separator=" ", strip=True)
        if text:
            parts.append(text)
    return " ".join(parts).strip()


def extract_in_brief(soup: BeautifulSoup) -> dict:
    """Extract the 'In Brief' section (Goal / What to do / Why important)."""
    brief = {}
    section = soup.find(id="brief")
    if not section:
        return brief
    for li in section.find_all("li") if section.name != "section" else []:
        text = li.get_text(separator=" ", strip=True)
        if "Goal" in text:
            brief["goal"] = text.replace("Goal", "").strip()
        elif "What to do" in text:
            brief["what_to_do"] = text.replace("What to do", "").strip()
        elif "Why" in text:
            brief["why_important"] = text.replace("Why it's important", "").strip()
    # Also try dl/dt/dd pattern
    for strong in (section if section else soup).find_all("strong"):
        label = strong.get_text(strip=True).lower()
        value = strong.next_sibling
        if value:
            value_text = value.strip() if isinstance(value, str) else value.get_text(strip=True)
        else:
            value_text = ""
        if "goal" in label:
            brief["goal"] = value_text
        elif "what to do" in label:
            brief["what_to_do"] = value_text
        elif "why" in label:
            brief["why_important"] = value_text
    return brief


def extract_benefits(soup: BeautifulSoup) -> list[str]:
    """Extract the Benefits section as a list of benefit descriptions."""
    benefits = []
    section = soup.find(id="benefits")
    if not section:
        return benefits
    for li in section.find_next("ul", recursive=False) or []:
        if hasattr(li, "get_text"):
            text = li.get_text(separator=" ", strip=True)
            if text:
                benefits.append(text)
    if not benefits:
        text = extract_section_text(soup, "benefits")
        if text:
            benefits.append(text)
    return benefits


def extract_examples(soup: BeautifulSoup) -> list[dict]:
    """Extract the Examples section."""
    examples = []
    section = soup.find(id="examples")
    if not section:
        return examples
    for item in section.find_next_siblings():
        if item.name and re.match(r"^h[2-3]$", item.name):
            break
        if item.name == "ul" or item.name == "ol":
            for li in item.find_all("li", recursive=False):
                strong = li.find("strong")
                title = strong.get_text(strip=True) if strong else ""
                desc = li.get_text(separator=" ", strip=True)
                if strong:
                    desc = desc.replace(title, "", 1).strip()
                if desc:
                    examples.append({"title": title, "description": desc})
    return examples


def extract_techniques(soup: BeautifulSoup) -> dict:
    """Extract Sufficient, Advisory, and Failure techniques."""
    result = {"sufficient": [], "advisory": [], "failures": []}

    for a_tag in soup.find_all("a", href=re.compile(r"/Techniques/")):
        href = a_tag.get("href", "")
        text = a_tag.get_text(separator=" ", strip=True)
        if not text or not href:
            continue

        # Extract technique ID from URL (e.g., G94, ARIA6, H37, F3)
        match = re.search(r"/([A-Z]+\d+)/?$", href)
        tech_id = match.group(1) if match else ""

        # Determine technology from ID prefix
        tech_prefix = re.match(r"^([A-Z]+)", tech_id).group(1) if tech_id else ""
        tech_type_map = {
            "G": "general", "H": "html", "C": "css", "ARIA": "aria",
            "PDF": "pdf", "F": "failure", "SCR": "script",
            "SM": "smil", "SVR": "server",
        }
        technology = tech_type_map.get(tech_prefix, "general")

        entry = {
            "id": tech_id,
            "title": text,
            "url": href if href.startswith("http") else urljoin("https://www.w3.org/", href),
            "technology": technology,
        }

        # Classify: failures start with F, rest depends on section context
        if tech_id.startswith("F"):
            result["failures"].append(entry)
        else:
            # Check parent context for sufficient vs advisory
            parent_section = a_tag.find_parent(["section", "div"])
            parent_text = ""
            if parent_section:
                prev_heading = parent_section.find(re.compile(r"^h[3-4]$"))
                parent_text = prev_heading.get_text() if prev_heading else ""
            if "advisory" in parent_text.lower():
                result["advisory"].append(entry)
            else:
                result["sufficient"].append(entry)

    # Deduplicate by id
    for key in result:
        seen = set()
        deduped = []
        for t in result[key]:
            if t["id"] not in seen:
                seen.add(t["id"])
                deduped.append(t)
        result[key] = deduped

    return result


def extract_test_rules(soup: BeautifulSoup) -> list[dict]:
    """Extract ACT Test Rules."""
    rules = []
    section = soup.find(id="test-rules")
    if not section:
        return rules
    for a_tag in section.find_next_siblings():
        if a_tag.name and re.match(r"^h[2-3]$", a_tag.name):
            break
        for link in (a_tag.find_all("a") if hasattr(a_tag, "find_all") else []):
            href = link.get("href", "")
            text = link.get_text(strip=True)
            if href and "act/rules" in href:
                rules.append({"title": text, "url": href})
    return rules


def extract_technologies(soup: BeautifulSoup, techniques: dict) -> list[str]:
    """Determine applicable technologies from technique set."""
    techs = set()
    for category in techniques.values():
        for t in category:
            tech = t.get("technology", "")
            if tech and tech != "general" and tech != "failure":
                techs.add(tech)
    if not techs:
        techs.add("html")  # default applicability
    return sorted(techs)


def classify_input_types(ref_id: str, description: str) -> list[str]:
    """Determine which input modalities are affected."""
    types = set()
    desc_lower = description.lower()

    # Explicit guideline-level mapping
    guideline = ".".join(ref_id.split(".")[:2])
    if guideline == "2.1":
        types.add("keyboard")
    elif guideline == "2.5":
        types.update(["pointer", "touch", "gesture"])

    # Keyword detection
    keyword_map = {
        "keyboard": ["keyboard", "focus", "tab", "key shortcut"],
        "pointer": ["pointer", "click", "mouse", "drag", "target size"],
        "touch": ["touch", "gesture", "swipe", "pinch"],
        "voice": ["voice", "speech"],
        "visual": ["color", "contrast", "visual", "text", "image", "visible", "reflow"],
        "auditory": ["audio", "captions", "sound", "sign language"],
    }
    for input_type, keywords in keyword_map.items():
        if any(kw in desc_lower for kw in keywords):
            types.add(input_type)

    return sorted(types) if types else ["visual"]


def get_wcag_version(ref_id: str) -> str:
    """Determine which WCAG version introduced this criterion."""
    if ref_id in WCAG_22_NEW:
        return "2.2"
    elif ref_id in WCAG_21_NEW:
        return "2.1"
    else:
        return "2.0"


def get_automatable(ref_id: str) -> str:
    """Classify how automatable the criterion is."""
    if ref_id in FULLY_AUTOMATABLE:
        return "full"
    elif ref_id in PARTIALLY_AUTOMATABLE:
        return "partial"
    else:
        return "manual"


def get_disability_impact(ref_id: str) -> list[str]:
    """Get disability categories impacted by this criterion."""
    guideline = ".".join(ref_id.split(".")[:2])
    return DISABILITY_IMPACT_MAP.get(guideline, ["general"])


def extract_related_scs(soup: BeautifulSoup, own_ref_id: str) -> list[str]:
    """Find cross-referenced success criteria from the Intent section."""
    related = set()
    intent_section = soup.find(id="intent")
    if not intent_section:
        return []
    for a_tag in intent_section.find_next_siblings():
        if a_tag.name and re.match(r"^h[2]$", a_tag.name):
            break
        for link in (a_tag.find_all("a") if hasattr(a_tag, "find_all") else []):
            href = link.get("href", "")
            # Match WCAG spec anchors or Understanding page URLs
            sc_match = re.search(r"#[\w-]+$", href)
            understanding_match = re.search(r"/Understanding/([\w-]+)\.html", href)
            text = link.get_text(strip=True)
            # Look for SC pattern in text like "1.4.3" or "Success Criterion 2.4.7"
            sc_pattern = re.search(r"\b(\d+\.\d+\.\d+)\b", text)
            if sc_pattern:
                found_id = sc_pattern.group(1)
                if found_id != own_ref_id:
                    related.add(found_id)
    return sorted(related)


def enrich_criterion(sc: dict, principle_title: str, guideline_title: str) -> dict:
    """Enrich a single success criterion with data from its Understanding page."""
    ref_id = sc["ref_id"]
    understanding_url = get_understanding_url(sc["url"])

    log.info("  Enriching %s: %s", ref_id, sc["title"])

    soup = fetch_page(understanding_url)

    # ── Data we can derive without scraping ──
    sc["wcag_version"] = get_wcag_version(ref_id)
    sc["automatable"] = get_automatable(ref_id)
    sc["disability_impact"] = get_disability_impact(ref_id)
    sc["input_types_affected"] = classify_input_types(ref_id, sc.get("description", ""))

    if soup:
        # ── Data from the Understanding page ──
        sc["in_brief"] = extract_in_brief(soup)
        sc["intent"] = extract_section_text(soup, "intent")[:2000]  # cap length
        sc["benefits"] = extract_benefits(soup)
        sc["examples"] = extract_examples(soup)
        sc["techniques"] = extract_techniques(soup)
        sc["test_rules"] = extract_test_rules(soup)
        sc["technology_applicability"] = extract_technologies(soup, sc["techniques"])
        sc["related_scs"] = extract_related_scs(soup, ref_id)

        log.info("    ✅ %d sufficient, %d advisory, %d failures, %d test rules",
                 len(sc["techniques"]["sufficient"]),
                 len(sc["techniques"]["advisory"]),
                 len(sc["techniques"]["failures"]),
                 len(sc["test_rules"]))
    else:
        # Populate empty structures so the schema is consistent
        sc["in_brief"] = {}
        sc["intent"] = ""
        sc["benefits"] = []
        sc["examples"] = []
        sc["techniques"] = {"sufficient": [], "advisory": [], "failures": []}
        sc["test_rules"] = []
        sc["technology_applicability"] = []
        sc["related_scs"] = []
        log.warning("    ⚠️  Could not fetch Understanding page")

    time.sleep(RATE_LIMIT_SECONDS)
    return sc


def main():
    log.info("=" * 62)
    log.info("  WCAG 2.2 JSON Enrichment")
    log.info("=" * 62)

    # ── Load base JSON ──
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    log.info("Loaded %d principles from %s", len(data), INPUT_FILE)

    total_criteria = 0

    for principle in data:
        log.info("\nPrinciple %s: %s", principle["ref_id"], principle["title"])

        for guideline in principle.get("guidelines", []):
            log.info("  Guideline %s: %s", guideline["ref_id"], guideline["title"])

            for sc in guideline.get("success_criteria", []):
                enrich_criterion(sc, principle["title"], guideline["title"])
                total_criteria += 1

    # ── Write enriched output ──
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

    log.info("\n" + "=" * 62)
    log.info("  Enriched %d criteria → %s", total_criteria, OUTPUT_FILE)
    log.info("=" * 62)


if __name__ == "__main__":
    main()
