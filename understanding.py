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
        ("draft_notice", r"\b(draft|prepare|compose).{0,20}\b(notice|legal\s*notice)\b", 0.9),
        ("find_judgements", r"\b(find|search|relevant|latest).{0,20}\b(judg(e)?ment|precedent|case\s*law|authorities)\b", 0.85),
        ("citation_finder", r"\b(citation|cite|reported\s*as|AIR|SCC|SCR|Cri\s*LJ)\b", 0.85),
        ("case_tracking", r"\b(track|tracking|status|next\s*hearing|hearing\s*date|case\s*status)\b", 0.8),
    ]
    generic_patterns = [
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
