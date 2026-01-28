from __future__ import annotations
import re
from typing import Dict, List, Optional, Tuple

ROLE_STUDENT = "student"
ROLE_PROFESSIONAL = "professional"


def classify_role(text: str) -> Optional[str]:
    """Extract role from free-form text. Returns 'student', 'professional', or None."""
    t = text.lower()
    # Prioritize explicit 'student'
    if re.search(r"\b(student|law\s*student|college|university|semester|moot)\b", t):
        return ROLE_STUDENT
    if re.search(r"\b(professional|advocate|lawyer|attorney|counsel|practitioner|firm|barrister)\b", t):
        return ROLE_PROFESSIONAL
    return None


def _detect_intent(text: str, role: Optional[str]) -> Tuple[str, float]:
    t = text.lower()
    score = 0.5
    # Student intents
    patterns_student = [
        ("pyqs", r"\b(pyq|previous\s*year|question\s*bank|practice\s*questions|mcq)\b", 0.9),
        ("case_summaries", r"\b(summary|summaries|brief|overview|digest)\b", 0.85),
        ("simplify_section", r"\b(section\s*\d+|explain\s*section|simplify\s*section|what\s*does\s*section)\b", 0.9),
        ("mock_argument", r"\b(moot|oral\s*argument|mock\s*argument|skeleton\s*argument)\b", 0.8),
    ]
    # Professional intents
    patterns_prof = [
        ("draft_notice", r"\b(draft|prepare|compose|create|write|make).{0,30}\b(notice|affidavit|agreement|contract|deed|will|application|petition|plaint|reply|statement|memo|opinion|lease|rent)\b", 0.9),
        ("find_judgements", r"\b(find|search|relevant|latest).{0,20}\b(judg(e)?ment|precedent|case\s*law|authorities)\b", 0.85),
        ("citation_finder", r"\b(citation|cite|reported\s*as|AIR|SCC|SCR|Cri\s*LJ)\b", 0.85),
        ("case_tracking", r"\b(track|tracking|status|next\s*hearing|hearing\s*date|case\s*status)\b", 0.8),
    ]
    generic_patterns = [
        # Queries asking about the objective/purpose/scope of a named Act
        # (e.g. "What is the objective of the Railway Property (Unlawful Possession) Act, 1966?")
        (
            "simplify_section",
            r"\b(objective|object|purpose|scope|aim)\b.{0,80}\b[ a-z()/-]+ act,?\s*(19|20)\d{2}\b",
            0.85,
        ),
        ("simplify_section", r"\b(section\s*\d+|ipc|crpc|evidence\s*act|contract\s*act|it\s*act|constitution\s*article)\b", 0.7),
        ("find_judgements", r"\b(case|judg(e)?ment|precedent|authority|ruling|decision)\b", 0.6),
    ]

    chosen = "legal_query"

    def match_patterns(patterns):
        nonlocal chosen, score
        for intent, pat, s in patterns:
            if re.search(pat, t):
                chosen = intent
                score = s
                return True
        return False

    if role == ROLE_STUDENT:
        if not match_patterns(patterns_student):
            match_patterns(generic_patterns)
    elif role == ROLE_PROFESSIONAL:
        if not match_patterns(patterns_prof):
            match_patterns(generic_patterns)
    else:
        # Unknown role, try both then generic
        if not match_patterns(patterns_student):
            if not match_patterns(patterns_prof):
                match_patterns(generic_patterns)

    return chosen, score


def _detect_topic(text: str) -> Tuple[str, float]:
    t = text.lower()
    topics = [
        ("criminal", r"\b(ipc|crpc|bail|fir|charge|section\s*302|section\s*420|theft|murder|assault|cheating|arrest)\b", 0.85),
        ("contract", r"\b(contract|breach|consideration|offer|acceptance|specific\s*performance|indemnity|guarantee)\b", 0.8),
        ("property", r"\b(property|title|possession|easement|mortgage|lease|tenancy|land|transfer\s*of\s*property)\b", 0.75),
        ("constitutional", r"\b(article\s*\d+|fundamental\s*right|writ|habeas|mandamus|constitution|basic\s*structure)\b", 0.8),
        ("family", r"\b(divorce|maintenance|custody|alimony|marriage|hindu\s*marriage|special\s*marriage|domestic\s*violence)\b", 0.75),
        ("tax", r"\b(gst|income\s*tax|vat|excise|customs|assessment|refund|input\s*credit)\b", 0.75),
        ("labour", r"\b(labour|labor|wages|termination|gratuity|pf|esi|industrial\s*dispute)\b", 0.7),
        ("civil", r"\b(civil\s*suit|injunction|damages|negligence|tort)\b", 0.65),
    ]
    for name, pat, s in topics:
        if re.search(pat, t):
            return name, s
    return "general", 0.5


def _extract_missing_fields(text: str, intent: str) -> List[str]:
    t = text.lower()
    fields_common = [
        "jurisdiction/court",
        "timeframe/date",
        "location/state",
        "facts summary",
    ]
    fields_by_intent = {
        "pyqs": ["subject/course", "semester"],
        "case_summaries": ["case name", "citation"],
        "simplify_section": ["act name", "section number"],
        "mock_argument": ["issue statement", "opponent position"],
        "draft_notice": ["parties", "relief sought", "cause of action"],
        "find_judgements": ["topic/section", "jurisdiction preference"],
        "citation_finder": ["case name", "reporter/series"],
        "case_tracking": ["court name", "case number"],
        "legal_query": ["topic/section", "desired outcome"],
    }
    fields = fields_common + fields_by_intent.get(intent, [])

    present_cues = {
        "jurisdiction/court": r"\b(supreme\s*court|high\s*court|district\s*court|sessions\s*court|magistrate)\b",
        "timeframe/date": r"\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\b20\d{2}\b|\b19\d{2}\b|yesterday|today|tomorrow)\b",
        "location/state": r"\b(delhi|mumbai|maharashtra|karnataka|up|uttar\s*pradesh|tamil\s*nadu|kerala|bengal|kolkata|chennai|bangalore)\b",
        "facts summary": r"\b(my|our|client|party|employer|tenant|landlord|police|complaint|agreement|contract)\b",
        "subject/course": r"\b(contract|ipc|constitution|torts?|property|evidence|company|family)\b",
        "semester": r"\b(sem|semester|year\s*\d|1st|2nd|3rd|4th|5th)\b",
        "case name": r"\b(v\.?\s|vs\.?\s|versus)\b",
        "citation": r"\b(air|scc|scr|all\s*er|wlr|manu)\b",
        "act name": r"\b(act|code|rules|regulation)\b",
        "section number": r"\bsection\s*\d+[a-zA-Z]*\b",
        "issue statement": r"\b(issue|question\s*of\s*law)\b",
        "opponent position": r"\b(opponent|respondent|defendant|prosecution|appellant)\b",
        "parties": r"\b(plaintiff|defendant|petitioner|respondent|applicant|accused|complainant)\b",
        "relief sought": r"\b(relief|prayer|seek|damages|compensation|injunction|declaration|quash)\b",
        "cause of action": r"\b(cause\s*of\s*action|breach|offence|injury|grievance)\b",
        "topic/section": r"\b(section\s*\d+|article\s*\d+|under\s*section|topic)\b",
        "jurisdiction preference": r"\b(supreme|high\s*court|state|national)\b",
        "reporter/series": r"\b(air|scc|scr|cri\s*lj|all\s*er|wlr)\b",
        "court name": r"\b(supreme|high\s*court|district|sessions|magistrate)\b",
        "case number": r"\b(\d{1,4}\/\d{2,4}|case\s*no\.?\s*\d+)\b",
        "desired outcome": r"\b(want|need|seek|aim|goal|objective|remedy)\b",
    }

    missing = []
    for f in fields:
        pat = present_cues.get(f)
        if pat and re.search(pat, t):
            continue
        missing.append(f)
    return missing


def analyze_query(text: str, role: Optional[str] = None) -> Dict:
    """Return HITL analysis JSON as required by spec."""
    intent, ic = _detect_intent(text, role)
    topic, tc = _detect_topic(text)
    missing = _extract_missing_fields(text, intent)

    # context completeness heuristic
    # Count how many of the first four common fields are present
    common = [
        "jurisdiction/court",
        "timeframe/date",
        "location/state",
        "facts summary",
    ]
    present = 0
    t = text.lower()
    present_cues_common = {
        "jurisdiction/court": r"\b(supreme\s*court|high\s*court|district\s*court|sessions\s*court|magistrate)\b",
        "timeframe/date": r"\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\b20\d{2}\b|\b19\d{2}\b|yesterday|today|tomorrow)\b",
        "location/state": r"\b(delhi|mumbai|maharashtra|karnataka|up|uttar\s*pradesh|tamil\s*nadu|kerala|bengal|kolkata|chennai|bangalore)\b",
        "facts summary": r"\b(my|our|client|party|employer|tenant|landlord|police|complaint|agreement|contract)\b",
    }
    for f in common:
        if re.search(present_cues_common[f], t):
            present += 1
    completeness = present / len(common)

    return {
        "intent": intent,
        "topic": topic,
        "missing_fields": missing,
        "intent_confidence": round(float(ic), 2),
        "topic_confidence": round(float(tc), 2),
        "context_completeness": round(float(completeness), 2),
    }


def parse_case_identity(text: str) -> Dict[str, Optional[str]]:
    """Strict extraction of case identity from user query.

    Only extracts fields when they appear *explicitly* in the text. No inference.

    Extracted fields (when present):
    - case_type:       e.g. "Criminal Revision Application"
    - case_number:     e.g. "2"
    - filing_year:     e.g. "2002" (for patterns like "No. 2 of 2002")
    - decision_date:   e.g. "1 July 2005" or "01/07/2005" (as string)
    - decision_year:   e.g. "2005" (from the decision_date)
    - court_name:      e.g. "Bombay High Court"
    - court_level:     "High Court" or "Supreme Court" if explicitly mentioned

    Also returns:
    - raw_string: the exact "<TYPE> No. X of YYYY" phrase if matched
    """

    t = text
    tl = text.lower()

    # 1. Court level / name (only from explicit court phrases)
    court_level: Optional[str] = None
    court_name: Optional[str] = None

    if "supreme court" in tl:
        court_level = "Supreme Court"
        court_name = "Supreme Court of India"
    elif "high court" in tl:
        court_level = "High Court"

        # Try common High Court header forms, e.g.
        # "HIGH COURT OF JUDICATURE AT BOMBAY" or "HIGH COURT AT BOMBAY"
        m = re.search(r"high court(?: of judicature)? at ([A-Za-z ]+)", tl, re.IGNORECASE)
        if m:
            place = m.group(1).strip().title()
            court_name = f"{place} High Court"

        # Fallback: "Bombay High Court" style, but only a single-word place
        # to avoid capturing full phrases like "dismissed for default by the Bombay".
        if not court_name:
            m2 = re.search(r"\b([A-Za-z]+)\s+high court\b", tl, re.IGNORECASE)
            if m2:
                place = m2.group(1).strip().title()
                court_name = f"{place} High Court"

    # 2. Case type / number / filing year patterns (no inference)
    raw_string: Optional[str] = None
    case_type: Optional[str] = None
    case_number: Optional[str] = None
    filing_year: Optional[str] = None

    # Strict HC-style pattern: "Criminal Revision Application No. 2 of 2002", etc.
    # Only allow known case-type tokens to avoid grabbing free text like "Why was ...".
    case_type_pattern = (
        r"Criminal\s+Revision\s+Application|"
        r"Civil\s+Revision\s+Application|"
        r"Criminal\s+Application|"
        r"Civil\s+Application|"
        r"Writ\s+Petition|"
        r"Criminal\s+Appeal|"
        r"Civil\s+Appeal"
    )

    patterns = [
        rf"(?P<case_type>{case_type_pattern})\s+No\.?\s*(?P<case_number>\d+)\s+of\s+(?P<filing_year>\d{{4}})",
        rf"(?P<case_type>{case_type_pattern})\s+No\.?\s*(?P<case_number>\d+)\s*/\s*(?P<filing_year>\d{{4}})",
        r"case\s+no\.?\s*(?P<case_number>\d+)[/\s]*(?P<filing_year>\d{4})",
    ]

    for pat in patterns:
        m = re.search(pat, t, re.IGNORECASE)
        if m:
            raw_string = m.group(0).strip()
            case_number = m.groupdict().get("case_number")
            fy = m.groupdict().get("filing_year")
            filing_year = fy if fy else None

            ct = m.groupdict().get("case_type")
            if ct:
                case_type = " ".join(ct.split()).strip()
            break

    # 3. Decision date / year (explicit only, no inference)
    decision_date: Optional[str] = None
    decision_year: Optional[str] = None

    # Pattern: "on 1 July 2005"
    m_date1 = re.search(r"\bon\s+(\d{1,2})\s+([A-Za-z]+)\s+(\d{4})", t, re.IGNORECASE)
    if m_date1:
        decision_date = m_date1.group(0).strip()
        decision_year = m_date1.group(3)
    else:
        # Pattern: "dated 01.07.2005" or "dated 01/07/2005"
        m_date2 = re.search(r"\bdated\s+(\d{1,2})[./-](\d{1,2})[./-](\d{2,4})", t, re.IGNORECASE)
        if m_date2:
            decision_date = m_date2.group(0).strip()
            y = m_date2.group(3)
            # Normalize 2-digit year approximately to 4-digit when obvious (e.g. '05' -> '2005')
            decision_year = y if len(y) == 4 else None

    return {
        "court_name": court_name,
        "court_level": court_level,
        "case_type": case_type,
        "case_number": case_number,
        "filing_year": filing_year,
        "decision_date": decision_date,
        "decision_year": decision_year,
        "raw_string": raw_string,
    }
