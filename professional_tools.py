from __future__ import annotations
from typing import List, Dict


def _format_irac(issue: str, rules: List[str], application: List[str], conclusion: str) -> str:
    rules_txt = "\n- ".join(["- "+r for r in rules]) if rules else "-"
    app_txt = "\n- ".join(["- "+a for a in application]) if application else "-"
    return (
        f"Issue:\n{issue}\n\n"
        f"Rule:\n{rules_txt}\n\n"
        f"Application:\n{app_txt}\n\n"
        f"Conclusion:\n{conclusion}"
    )


def _case_refs(retrieved: List[Dict]) -> List[str]:
    return [f"{c['title']} ({c['source'].title()} Ct.)" for c in retrieved]


def draft_notice(query: str, topic: str, retrieved: List[Dict]) -> str:
    issue = "Draft a concise legal notice capturing cause of action and relief sought."
    rules = ["Essential elements of a legal notice under applicable law", *_case_refs(retrieved)]
    application = [
        "Facts: Briefly narrate material facts and breach/offence.",
        "Demand: Specify performance or cessation expected within a reasonable time.",
        "Consequence: Indicate intended action upon non-compliance.",
    ]
    conclusion = "This structure can be adapted to your facts; review before dispatch."
    return _format_irac(issue, rules, application, conclusion)


def find_judgements(query: str, topic: str, retrieved: List[Dict]) -> str:
    issue = f"Identify leading judgements relevant to {topic}."
    rules = _case_refs(retrieved)
    application = [f"Why relevant: {c['summary']}" for c in retrieved]
    conclusion = "Use these as starting points; Shepardize/KeyCite in official databases."
    return _format_irac(issue, rules, application, conclusion)


def citation_finder(query: str, topic: str, retrieved: List[Dict]) -> str:
    issue = "Surface likely reporter citations based on the case/topic cues."
    rules = ["Common Indian reporters: AIR, SCC, SCR, Cri LJ", *_case_refs(retrieved)]
    application = [
        "Cross-check party names and year in authentic databases.",
        "Prefer neutral citations where available.",
    ]
    conclusion = "Confirm final citations from official reporters or court websites."
    return _format_irac(issue, rules, application, conclusion)


def case_tracking(query: str, topic: str, retrieved: List[Dict]) -> str:
    issue = "Outline steps to track the case status and upcoming hearings."
    rules = [
        "Use e-Courts/Supreme Court/High Court portals with case/party numbers",
        *_case_refs(retrieved),
    ]
    application = [
        "Check cause list and order sheets.",
        "Set calendar reminders for dates; maintain client updates.",
    ]
    conclusion = "Automate reminders and verify status a day prior to hearing."
    return _format_irac(issue, rules, application, conclusion)
