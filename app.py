from __future__ import annotations
import os
import streamlit as st
from typing import Dict, List, Optional
from pathlib import Path
from dotenv import load_dotenv

# Prefer absolute imports so the app runs with `streamlit run ai_lawyer/app.py`
import understanding
import rag
import student_tools
import professional_tools

APP_TITLE = "LexiCounsel — AI Lawyer MVP"
GREETING = "Hi! Are you a Student or a Professional?"

# Load .env from this package folder (ai_lawyer/.env) if present, without overriding
# any existing environment variables set in the shell or host environment.
try:
    load_dotenv(dotenv_path=Path(__file__).with_name(".env"), override=False)
    # Optional: a tiny console hint (no secrets logged)
    if os.getenv("GEMINI_API_KEY"):
        print("[AI-LAWYER] .env loaded; GEMINI_API_KEY present in environment.")
    else:
        print("[AI-LAWYER] .env checked; GEMINI_API_KEY not found in environment.")
except Exception as _e:
    # Fail silently; we'll still attempt st.secrets as a fallback inside _get_gemini_api_key
    print(f"[AI-LAWYER] .env load skipped/failed: {_e}")


def _init_session():
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    if "role" not in st.session_state:
        st.session_state["role"] = None
    if "clarify_rounds" not in st.session_state:
        st.session_state["clarify_rounds"] = 0
    if "last_assistant_type" not in st.session_state:
        st.session_state["last_assistant_type"] = None
    if not st.session_state["messages"]:
        # First bot message
        st.session_state["messages"].append({"role": "assistant", "content": GREETING})


def _render_history():
    for m in st.session_state["messages"]:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])


def _get_gemini_api_key() -> Optional[str]:
    """Safely fetch GEMINI_API_KEY from env or Streamlit secrets without raising.

    Avoids touching st.secrets unless needed and wraps access to prevent
    FileNotFoundError when no secrets.toml file exists.
    """
    key = os.getenv("GEMINI_API_KEY")
    if key:
        return key
    try:
        # Accessing st.secrets can raise if no secrets file exists
        return st.secrets.get("GEMINI_API_KEY")  # type: ignore[attr-defined]
    except Exception:
        return None


def _final_confidence(analysis: Dict) -> float:
    return round(
        (
            float(analysis["intent_confidence"]) +
            float(analysis["topic_confidence"]) +
            float(analysis["context_completeness"]) 
        ) / 3.0,
        2,
    )


def _clarifying_prompt(missing_fields: List[str]) -> str:
    if not missing_fields:
        return "Could you please share a few more details (jurisdiction, timeframe, and a brief fact summary)?"
    qs = []
    for f in missing_fields[:3]:
        if f == "jurisdiction/court":
            qs.append("Which court or jurisdiction does this relate to (e.g., Supreme Court, High Court, state)?")
        elif f == "timeframe/date":
            qs.append("What timeframe or relevant dates should I consider?")
        elif f == "location/state":
            qs.append("Which city/state or location is involved?")
        elif f == "facts summary":
            qs.append("Could you share a brief summary of key facts?")
        elif f == "case name":
            qs.append("Do you have the case name or party names?")
        elif f == "section number":
            qs.append("Which statutory section or article should I focus on?")
        elif f == "relief sought":
            qs.append("What relief are you seeking?")
        else:
            qs.append(f"Could you provide: {f}?")
    return "To guide you better, a few quick questions:\n- " + "\n- ".join(qs)


def _route_and_answer(role: str, intent: str, query: str, topic: str, retrieved: List[Dict]) -> str:
    # Student tool routing
    if role == understanding.ROLE_STUDENT:
        if intent == "pyqs":
            return student_tools.pyqs(query, topic, retrieved)
        if intent == "case_summaries":
            return student_tools.case_summaries(query, topic, retrieved)
        if intent == "simplify_section":
            return student_tools.simplify_section(query, topic, retrieved)
        if intent == "mock_argument":
            return student_tools.mock_argument(query, topic, retrieved)
        # Fallback generic answer for students
        return student_tools.case_summaries(query, topic, retrieved)

    # Professional tool routing
    if role == understanding.ROLE_PROFESSIONAL:
        if intent == "draft_notice":
            return professional_tools.draft_notice(query, topic, retrieved)
        if intent == "find_judgements":
            return professional_tools.find_judgements(query, topic, retrieved)
        if intent == "citation_finder":
            return professional_tools.citation_finder(query, topic, retrieved)
        if intent == "case_tracking":
            return professional_tools.case_tracking(query, topic, retrieved)
        # Fallback generic answer for professionals
        return professional_tools.find_judgements(query, topic, retrieved)

    # Unknown role fallback (should be rare)
    return student_tools.case_summaries(query, topic, retrieved)


def _llm_irac_answer(query: str, role: str, intent: str, topic: str, retrieved: List[Dict]) -> Optional[str]:
    """Use LangChain + Gemini to generate the final IRAC answer when configured.

    Returns None on failure so the caller can fall back to deterministic tools.
    """
    # Try environment variable, then Streamlit secrets if available (safely)
    api_key = _get_gemini_api_key()
    if not api_key:
        return None

    cases_txt = "\n".join([f"- {c['title']}: {c['summary']}" for c in (retrieved or [])])
    prompt = f"""
You are a helpful legal assistant. Write the answer strictly in IRAC format with clear headings:
Issue:\nRule:\nApplication:\nConclusion:

Do not display confidence scores, internal reasoning, or tool names. Keep language appropriate for the user's role.

Role: {role}
Intent: {intent}
Topic: {topic}
User query: "{query}"

Relevant cases (for context):
{cases_txt if cases_txt else '- (no specific cases)'}

Produce a self-contained IRAC answer that is concise, accurate, and avoids revealing internal logic.
"""
    # Prefer LangChain GoogleGenAI; fall back to direct SDK if LC not available
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=api_key,
            temperature=0.2,
        )
        print("[AI-LAWYER] Gemini (LangChain) → Generating IRAC answer …")
        resp = llm.invoke(prompt)
        if isinstance(resp, str):
            text = resp.strip()
            _pv = text[:120].replace("\n", " ")
            print(f"[AI-LAWYER] Gemini (LangChain) ✓ IRAC chars={len(text)} preview={_pv}…")
            return text
        content = getattr(resp, "content", None)
        if isinstance(content, str):
            text = content.strip()
            _pv = text[:120].replace("\n", " ")
            print(f"[AI-LAWYER] Gemini (LangChain) ✓ IRAC chars={len(text)} preview={_pv}…")
            return text
        if isinstance(content, list):
            try:
                joined = "\n".join([(p.get("text") if isinstance(p, dict) else str(p)) for p in content]).strip()
                _pv = joined[:120].replace("\n", " ")
                print(f"[AI-LAWYER] Gemini (LangChain) ✓ IRAC chars={len(joined)} preview={_pv}…")
                return joined
            except Exception:
                print("[AI-LAWYER] Gemini (LangChain) × Unable to parse content list.")
                return None
        return None
    except Exception as e:
        print(f"[AI-LAWYER] Gemini (LangChain) × Exception: {e}")
        try:
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel('gemini-2.5-flash')
            print("[AI-LAWYER] Gemini (SDK) → Generating IRAC answer …")
            resp = model.generate_content(prompt)
            text = getattr(resp, 'text', None)
            if text:
                t = text.strip()
                _pv = t[:120].replace("\n", " ")
                print(f"[AI-LAWYER] Gemini (SDK) ✓ IRAC chars={len(t)} preview={_pv}…")
                return t
            print("[AI-LAWYER] Gemini (SDK) × Empty text returned.")
            return None
        except Exception as e2:
            print(f"[AI-LAWYER] Gemini (SDK) × Exception: {e2}")
            return None


def _llm_clarify(user_input: str, role: Optional[str], intent: str, topic: str, missing_fields: List[str]) -> Optional[str]:
    """Ask Gemini to craft concise clarifying questions in a polished, user-friendly template.

    Returns None if Gemini is not configured or fails.
    """
    api_key = _get_gemini_api_key()
    if not api_key:
        return None
    # Build a friendly task line and examples tailored to topic/intent
    friendly_task_by_intent = {
        "pyqs": "fetch {topic}-law PYQs for you",
        "case_summaries": "pull key case summaries for you",
        "simplify_section": "explain the section in plain language",
        "mock_argument": "help craft a short mock argument",
        "draft_notice": "draft a concise legal notice",
        "find_judgements": "surface leading judgements",
        "citation_finder": "find likely reporter citations",
        "case_tracking": "outline steps to track case status",
        "legal_query": "help with your query",
    }
    t = topic if topic and topic != "general" else "your"
    task_phrase = friendly_task_by_intent.get(intent, "help with your query").format(topic=t)

    topic_examples = {
        "criminal": "murder, offences against property, evidence",
        "contract": "offer & acceptance, consideration, breach, damages",
        "property": "ownership, possession, easement",
        "constitutional": "fundamental rights, writs, basic structure",
        "family": "divorce, maintenance, custody",
        "tax": "GST, input credit, assessment",
        "labour": "termination, wages, gratuity",
        "general": "key themes relevant to your query",
    }
    examples = topic_examples.get(topic or "general", topic_examples["general"])

    # Role-specific capabilities to show user what we can do for them
    caps_by_role = {
        "student": [
            "PYQs",
            "case summaries",
            "simplify sections",
            "mock arguments",
        ],
        "professional": [
            "draft notices",
            "find judgements",
            "citation finder",
            "case tracking",
        ],
        "unknown": [
            "PYQs / case summaries",
            "draft notices / find judgements",
            "citation finder / case tracking",
        ],
    }
    role_key = (role or "unknown").lower()
    role_key = role_key if role_key in caps_by_role else "unknown"
    caps_list = caps_by_role[role_key]
    caps_str = ", ".join(caps_list)

    # Helpful hints for LLM about what is missing (internal guidance)
    mf_txt = ", ".join(missing_fields[:6]) if missing_fields else "jurisdiction/court, timeframe/date, location/state"

    prompt = f"""
You are assisting a legal chatbot. Generate a polished clarifying prompt using this exact format and tone.

Constraints:
- Do not show internal reasoning, scores, or tool names.
- Start with: "For a {role_key}, I can help you with: {caps_str}. A few quick details will help:"
    (Use an em dash — only if you naturally need one; do not include "Great — I’ll …".)
- Then list exactly three bullets using the • character, each ending with a question mark.
- Bullets should be:
    • Jurisdiction (India / UK / US / other)?
    • Year or exam (e.g., 2019–2023, semester finals)?
    • Topics you want to focus on (e.g., {examples}) — or type ‘any’ for a broad mix?

Context for you (not to be shown to the user):
- Role: {role or 'unknown'}
- Detected intent: {intent}
- Topic: {topic}
- Missing info hints: {mf_txt}
- User message: "{user_input}"

Return only the formatted text described above, nothing else.
"""
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=api_key, temperature=0.2)
        print("[AI-LAWYER] Gemini (LangChain) → Generating clarifying questions …")
        resp = llm.invoke(prompt)
        if isinstance(resp, str):
            text = resp.strip()
            _pv = text[:120].replace("\n", " ")
            print(f"[AI-LAWYER] Gemini (LangChain) ✓ Clarify chars={len(text)} preview={_pv}…")
            return text
        content = getattr(resp, "content", None)
        if isinstance(content, str):
            text = content.strip()
            _pv = text[:120].replace("\n", " ")
            print(f"[AI-LAWYER] Gemini (LangChain) ✓ Clarify chars={len(text)} preview={_pv}…")
            return text
        if isinstance(content, list):
            try:
                joined = "\n".join([(p.get("text") if isinstance(p, dict) else str(p)) for p in content]).strip()
                _pv = joined[:120].replace("\n", " ")
                print(f"[AI-LAWYER] Gemini (LangChain) ✓ Clarify chars={len(joined)} preview={_pv}…")
                return joined
            except Exception:
                print("[AI-LAWYER] Gemini (LangChain) × Unable to parse content list.")
                return None
        return None
    except Exception as e:
        print(f"[AI-LAWYER] Gemini (LangChain) × Exception: {e}")
        try:
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel('gemini-2.5-flash')
            print("[AI-LAWYER] Gemini (SDK) → Generating clarifying questions …")
            resp = model.generate_content(prompt)
            text = getattr(resp, 'text', None)
            if text:
                t = text.strip()
                _pv = t[:120].replace("\n", " ")
                print(f"[AI-LAWYER] Gemini (SDK) ✓ Clarify chars={len(t)} preview={_pv}…")
                return t
            print("[AI-LAWYER] Gemini (SDK) × Empty text returned.")
            return None
        except Exception as e2:
            print(f"[AI-LAWYER] Gemini (SDK) × Exception: {e2}")
            return None


def main():
    st.set_page_config(page_title=APP_TITLE, page_icon="⚖️", layout="centered")
    _init_session()
    _render_history()

    user_input = st.chat_input("Type your message…")
    if not user_input:
        return

    # Add user message
    st.session_state["messages"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Role handling
    role_before = st.session_state.get("role")
    if role_before is None:
        detected = understanding.classify_role(user_input)
        if detected:
            st.session_state["role"] = detected
            # Immediately present role-specific capabilities and clarifying bullets via Gemini
            role_intro = _llm_clarify(
                user_input,
                detected,
                "legal_query",
                "general",
                [],
            )
            if not role_intro:
                role_intro = "I'm currently unavailable to proceed because the LLM isn't configured. Please set GEMINI_API_KEY."
            with st.chat_message("assistant"):
                st.markdown(role_intro)
            st.session_state["messages"].append({
                "role": "assistant",
                "content": role_intro,
            })
            st.session_state["clarify_rounds"] = 1
            st.session_state["last_assistant_type"] = "clarify"
            return
        else:
            # Ask for role again politely and stop
            with st.chat_message("assistant"):
                st.markdown("To personalize the assistance: Are you a Student or a Professional?")
            st.session_state["messages"].append({
                "role": "assistant",
                "content": "To personalize the assistance: Are you a Student or a Professional?",
            })
            return

    # Analyze query (HITL)
    role = st.session_state["role"]
    analysis = understanding.analyze_query(user_input, role)
    final_conf = _final_confidence(analysis)  # INTERNAL ONLY

    if final_conf < 0.75:
        # If we've already asked once, proceed with sensible defaults instead of looping
        if st.session_state.get("clarify_rounds", 0) >= 1:
            assumed = {
                "jurisdiction": "India",
                "years": "recent 5 years",
                "topics": (
                    analysis.get("topic")
                    if (analysis.get("topic") and analysis.get("topic") != "general")
                    else "core relevant topics"
                ),
            }
            # Proceed to answer below using these assumptions (passed via prompt augmentation)
        else:
            clarify = _llm_clarify(
                user_input,
                role,
                analysis.get("intent", "legal_query"),
                analysis.get("topic", "general"),
                analysis.get("missing_fields", []),
            )
            if not clarify:
                clarify = "I'm currently unavailable to proceed because the LLM isn't configured. Please set GEMINI_API_KEY."
            with st.chat_message("assistant"):
                st.markdown(clarify)
            st.session_state["messages"].append({"role": "assistant", "content": clarify})
            st.session_state["clarify_rounds"] = st.session_state.get("clarify_rounds", 0) + 1
            st.session_state["last_assistant_type"] = "clarify"
            return

    # Proceed: retrieve and answer (FAISS-connected)
    court_opt = analysis.get("court")  # optional from understanding
    if isinstance(court_opt, str):
        court_opt = court_opt.upper()
        court_opt = court_opt if court_opt in {"SC", "HC"} else None
    retrieved = rag.search_chunks(
        user_input,
        top_k=5,
        court=court_opt,
        year_min=analysis.get("year_min"),
        year_max=analysis.get("year_max"),
        doc_id_like=None,
    )
    intent = analysis.get("intent", "legal_query")
    topic = analysis.get("topic", "general")

    # Always use Gemini response; do not fall back to offline deterministic tools
    # If we had to assume defaults after a clarifier, include them in the prompt
    if "assumed" in locals():
        assumed_text = (
            f"\n\nAssumptions for drafting (not to display as scores):\n"
            f"- Jurisdiction: {assumed['jurisdiction']}\n"
            f"- Years: {assumed['years']}\n"
            f"- Topics: {assumed['topics']}\n"
        )
        enriched_query = user_input + assumed_text
        answer = _llm_irac_answer(enriched_query, role, intent, topic, retrieved)
    else:
        answer = _llm_irac_answer(user_input, role, intent, topic, retrieved)
    if not answer:
        answer = "I'm currently unavailable to proceed because the LLM isn't configured. Please set GEMINI_API_KEY."

    with st.chat_message("assistant"):
        st.markdown(answer)
    st.session_state["messages"].append({"role": "assistant", "content": answer})
    st.session_state["last_assistant_type"] = "final"


if __name__ == "__main__":
    main()
