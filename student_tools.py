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


def pyqs(query: str, topic: str, retrieved: List[Dict]) -> str:
    issue = f"Practice questions on {topic if topic != 'general' else 'the relevant subject area'} based on your prompt."
    rules = _case_refs(retrieved)
    application = [
        "Q1: Identify the key elements and apply them to a novel fact pattern.",
        "Q2: Briefly explain the applicable provision and one precedent.",
        "Q3: Draft a short IRAC for the scenario provided.",
    ]
    conclusion = "Use these to self-evaluate; compare with model answers from your course materials."
    return _format_irac(issue, rules, application, conclusion)


def case_summaries(query: str, topic: str, retrieved: List[Dict]) -> str:
    issue = f"Summarize key cases related to {topic}."
    rules = _case_refs(retrieved)
    application = [f"Summary: {c['summary']}" for c in retrieved]
    conclusion = "These summaries are for learning; verify with authentic case reporters."
    return _format_irac(issue, rules, application, conclusion)


def simplify_section(query: str, topic: str, retrieved: List[Dict]) -> str:
    issue = "Understand the meaning and scope of the mentioned statutory section/article."
    rules = ["Plain-language paraphrase of the section", *_case_refs(retrieved)]
    application = [
        "Break the section into elements.",
        "Illustrate with a simple example scenario.",
    ]
    conclusion = "This simplification aids learning; consult the bare act for precise wording."
    return _format_irac(issue, rules, application, conclusion)


def mock_argument(query: str, topic: str, retrieved: List[Dict]) -> str:
    issue = "Frame a short skeleton argument for a moot problem."
    rules = ["Applicable statute/test", *_case_refs(retrieved)]
    application = [
        "For Appellant: two concise submissions with authority.",
        "For Respondent: two concise counter-submissions with authority.",
    ]
    conclusion = "Conclude with the relief sought and why the test favors your side."
    return _format_irac(issue, rules, application, conclusion)
