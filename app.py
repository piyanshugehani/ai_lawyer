from __future__ import annotations
import os
import streamlit as st
from typing import Dict, List, Optional, Any
from pathlib import Path
from dotenv import load_dotenv

# Prefer absolute imports so the app runs with `streamlit run ai_lawyer/app.py`
import understanding
import rag
import student_tools
import professional_tools
try:
    from langchain.memory import ConversationBufferWindowMemory
    from langchain_core.messages import HumanMessage, AIMessage
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain.chains import ConversationChain
except Exception:
    ConversationBufferWindowMemory = None  # type: ignore
    HumanMessage = None  # type: ignore
    AIMessage = None  # type: ignore
    ChatGoogleGenerativeAI = None  # type: ignore
    ConversationChain = None  # type: ignore

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


def _env_flag(name: str, default: str = "0") -> bool:
    v = os.getenv(name, default)
    if v is None:
        return default == "1"
    s = str(v).strip().strip('"').strip("'").lower()
    return s in {"1", "true", "yes", "on"}


def _init_session():
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    if "role" not in st.session_state:
        st.session_state["role"] = None
    # Structured, silent user preferences memory
    if "user_prefs" not in st.session_state:
        st.session_state["user_prefs"] = {
            "role": None,                 # "student" | "professional"
            "jurisdiction": None,         # "India" | "UK" | "US" | other string
            "style": None,                # e.g., "no IRAC", "natural summaries", "bullet points"
            "exam_years": None,           # freeform string like "2019–2023"
            "topics": None,               # freeform string like "criminal, contract"
        }
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
    key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if key:
        return key
    try:
        # Accessing st.secrets can raise if no secrets file exists
        return st.secrets.get("GEMINI_API_KEY")  # type: ignore[attr-defined]
    except Exception:
        return None


def _get_gemini_model_name() -> str:
    """Return the generative model name (defaults to Gemini 2.5 Flash)."""
    v = os.getenv("GEMINI_GEN_MODEL", "gemini-2.5-flash")
    return str(v).strip().strip('"').strip("'") or "gemini-2.5-flash"


def _maybe_init_memory():
    """Initialize LangChain conversational memory if enabled and available."""
    use_mem = _env_flag("USE_LANGCHAIN_MEMORY", "1")
    if not use_mem or ConversationBufferWindowMemory is None:
        return None
    if "lc_memory" not in st.session_state:
        k = int(os.getenv("MEMORY_WINDOW", "6"))
        st.session_state["lc_memory"] = ConversationBufferWindowMemory(k=k, return_messages=True)
    return st.session_state.get("lc_memory")


def _maybe_init_lc_chain():
    """Initialize a LangChain ConversationChain with Gemini when enabled."""
    use_lc = _env_flag("USE_LANGCHAIN_GENAI", "0")
    if not use_lc or ChatGoogleGenerativeAI is None or ConversationChain is None:
        if not use_lc:
            print("[AI-LAWYER] LangChain generation disabled via USE_LANGCHAIN_GENAI.")
        elif ChatGoogleGenerativeAI is None or ConversationChain is None:
            print("[AI-LAWYER] LangChain packages unavailable; install langchain-google-genai, langchain-core.")
        return None
    if "lc_chain" in st.session_state and st.session_state["lc_chain"] is not None:
        return st.session_state["lc_chain"]
    api_key = _get_gemini_api_key()
    if not api_key:
        print("[AI-LAWYER] LangChain chain not created: missing GEMINI_API_KEY/GOOGLE_API_KEY.")
        return None
    model_name = _get_gemini_model_name()
    llm = ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=api_key,
        temperature=0.2,
        top_p=0.95,
        top_k=40,
        max_output_tokens=2048,
    )
    mem = _maybe_init_memory()
    chain = ConversationChain(llm=llm, memory=mem, verbose=False) if mem else ConversationChain(llm=llm, verbose=False)
    st.session_state["lc_chain"] = chain
    return chain


def _memory_add(role: str, content: str):
    mem = _maybe_init_memory()
    if not mem or not content:
        return
    try:
        if role == "user" and HumanMessage is not None:
            mem.chat_memory.add_message(HumanMessage(content=content))
        elif role == "assistant" and AIMessage is not None:
            mem.chat_memory.add_message(AIMessage(content=content))
    except Exception:
        pass


def _format_recent_history(max_chars: int = 1200) -> Optional[str]:
    mem = st.session_state.get("lc_memory")
    if not mem:
        # fallback to session messages last few turns
        msgs = st.session_state.get("messages", [])[-6:]
        if not msgs:
            return None
        lines = [f"{m['role'].capitalize()}: {m['content']}" for m in msgs if isinstance(m, dict) and m.get("content")]
        text = "\n".join(lines)
        return text[-max_chars:]
    try:
        vars = mem.load_memory_variables({})
        history = vars.get("history", [])
        if not history:
            return None
        lines = []
        for m in history:
            role = getattr(m, "type", None) or m.__class__.__name__
            if role.lower().startswith("human"):
                prefix = "User"
            elif role.lower().startswith("ai"):
                prefix = "Assistant"
            else:
                prefix = role
            lines.append(f"{prefix}: {getattr(m, 'content', '')}")
        text = "\n".join(lines)
        return text[-max_chars:]
    except Exception:
        return None


# ------------------------------
# Silent Preferences Memory
# ------------------------------
def _prefs() -> Dict[str, Optional[str]]:
    return st.session_state.get("user_prefs", {})


def _normalize_role(text: str) -> Optional[str]:
    import re
    t = text.lower()
    # Robust student detection: handles typos like "studnt", "stdnt"
    if re.search(r"\bstud\w*\b", t) or any(w in t for w in ["undergrad", "llb", "llm"]):
        return "student"
    # Professional detection: prefer explicit legal roles or clear prefix
    if re.search(r"\b(prof(essional)?|lawyer|advocate|attorney|practitioner)\b", t):
        return "professional"
    return None


def _extract_prefs_from_text(user_text: str) -> Dict[str, Optional[str]]:
    import re
    out: Dict[str, Optional[str]] = {}
    t = user_text.strip()
    tl = t.lower()
    # Guardrails: user asked not to store or says temporary
    if any(phrase in tl for phrase in ["don't save", "do not save", "dont save", "temporary", "temp only"]):
        return {}

    # Role
    role = _normalize_role(t)
    if role:
        out["role"] = role

    # Jurisdiction
    if re.search(r"\b(india|indian)\b", tl):
        out["jurisdiction"] = "India"
    elif re.search(r"\b(united\s*kingdom|uk|england|britain)\b", tl):
        out["jurisdiction"] = "UK"
    elif re.search(r"\b(united\s*states|us|usa|american)\b", tl):
        out["jurisdiction"] = "US"

    # Style preferences
    if "no irac" in tl or "avoid irac" in tl:
        out["style"] = "no IRAC"
    elif "irac" in tl and any(w in tl for w in ["use", "prefer", "format"]):
        out["style"] = "IRAC"
    elif any(phrase in tl for phrase in ["natural summary", "natural summaries", "simple summary", "plain english", "simple language"]):
        out["style"] = "natural summaries"
    elif any(phrase in tl for phrase in ["bullet points", "bulleted", "bullets"]):
        out["style"] = "bullet points"

    # Exam years (capture ranges or lists of years)
    m_range = re.search(r"(20\d{2})\s*[–-]\s*(20\d{2})", t)
    if m_range:
        out["exam_years"] = f"{m_range.group(1)}–{m_range.group(2)}"
    else:
        years = re.findall(r"\b(20\d{2})\b", t)
        if years and len(years) >= 1:
            out["exam_years"] = ", ".join(sorted(set(years)))

    # Topics / interest areas
    m_topics = re.search(r"(topics|interest areas|areas of interest)\s*[:\-]\s*(.+)$", t, flags=re.IGNORECASE)
    if m_topics:
        out["topics"] = m_topics.group(2).strip()

    return out


def _update_prefs_from_analysis(analysis: Dict[str, Any]):
    # Role
    role = analysis.get("role") or analysis.get("detected_role")
    if isinstance(role, str) and role.lower() in {"student", "professional"}:
        st.session_state["user_prefs"]["role"] = role.lower()
    # Topics
    topic = analysis.get("topic")
    if isinstance(topic, str) and topic and topic != "general":
        st.session_state["user_prefs"]["topics"] = topic


def _merge_prefs(new_bits: Dict[str, Optional[str]]):
    if not new_bits:
        return
    for k, v in new_bits.items():
        if v and isinstance(v, str):
            st.session_state["user_prefs"][k] = v


def _apply_style_to_query(query: str, intent: str) -> str:
    p = _prefs()
    style = p.get("style")
    if not style:
        return query
    # For case summaries or judgement finding, push style hint
    intents_for_style = {"case_summaries", "find_judgements", "simplify_section", "mock_argument"}
    if intent in intents_for_style:
        return query + f"\n\n[Style preference: {style} — apply in the response]"
    return query


def _jurisdiction_defaults_text() -> Optional[str]:
    p = _prefs()
    if p.get("jurisdiction"):
        return f"Assume jurisdiction: {p['jurisdiction']}."
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
    """Generate the final IRAC answer using Gemini.

    Defaults to the official SDK. Optionally uses LangChain wrapper when USE_LANGCHAIN_GENAI=1.
    """
    api_key = _get_gemini_api_key()
    if not api_key:
        return None

    cases_txt = "\n".join([f"- {c['title']}: {c['summary']}" for c in (retrieved or [])])
    user_prefs = _prefs()
    preferred_style = (user_prefs.get("style") or "").lower()
    j_text = _jurisdiction_defaults_text()
    history_txt = _format_recent_history() or ""
    history_block = f"Conversation (recent):\n{history_txt}\n\n" if history_txt else ""
    # Build a preferences block (hidden instruction)
    prefs_lines: List[str] = []
    if j_text:
        prefs_lines.append(j_text)
    if user_prefs.get("style"):
        prefs_lines.append(f"Apply user's style preference: {user_prefs['style']}.")
    if user_prefs.get("exam_years"):
        prefs_lines.append(f"Consider exam years: {user_prefs['exam_years']}.")
    if user_prefs.get("topics"):
        prefs_lines.append(f"Focus topics if relevant: {user_prefs['topics']}.")
    prefs_block = ("Internal constraints (not to display):\n" + "\n".join(prefs_lines) + "\n\n") if prefs_lines else ""

    # Choose style: IRAC (default) vs alternative styles
    use_irac = True
    bullet_mode = False
    if "no irac" in preferred_style or "natural" in preferred_style:
        use_irac = False
    if "bullet" in preferred_style:
        use_irac = False
        bullet_mode = True

    if use_irac:
        style_block = (
            "You are a helpful legal assistant. Write the answer strictly in IRAC format with clear headings:\n"
            "Issue:\nRule:\nApplication:\nConclusion:\n\n"
        )
        end_block = "Produce a self-contained IRAC answer that is concise, accurate, and avoids revealing internal logic."
    else:
        if bullet_mode:
            style_block = (
                "You are a helpful legal assistant. Provide a concise, accurate answer as clear bullet points.\n"
                "- Use plain language.\n- Avoid IRAC headings.\n- Keep it directly useful to the user.\n\n"
            )
        else:
            style_block = (
                "You are a helpful legal assistant. Provide a concise, natural-language legal summary.\n"
                "- Use plain language.\n- Avoid IRAC headings.\n- Be structured but not verbose.\n\n"
            )
        end_block = "Produce a self-contained answer that is accurate and avoids revealing internal logic."

    prompt = f"""
{history_block}{prefs_block}{style_block}
Do not display confidence scores, internal reasoning, or tool names. Keep language appropriate for the user's role.

Role: {role}
Intent: {intent}
Topic: {topic}
User query: "{query}"

Relevant cases (for context):
{cases_txt if cases_txt else '- (no specific cases)'}

{end_block}
"""

    use_lc = os.getenv("USE_LANGCHAIN_GENAI", "0") == "1"
    model_name = _get_gemini_model_name()

    # Preferred path: official SDK (stable across versions)
    if not use_lc:
        try:
            gen_cfg = {
                "temperature": 0.2,
                "top_p": 0.95,
                "top_k": 40,
                "max_output_tokens": 2048,
            }
            print("[AI-LAWYER] Gemini (SDK) → Generating answer …")
            try:
                # Attempt configured model, then sensible fallbacks
                candidates = [model_name, "gemini-2.0-flash-exp", "gemini-1.5-flash", "gemini-1.5-pro"]
                import google.generativeai as genai
                genai.configure(api_key=api_key)
                last_err = None
                for m in candidates:
                    try:
                        model = genai.GenerativeModel(m)
                        resp = model.generate_content(prompt, generation_config=gen_cfg)
                        text = getattr(resp, 'text', None)
                        if text and text.strip():
                            t = text.strip()
                            _pv = t[:120].replace("\n", " ")
                            print(f"[AI-LAWYER] Gemini (SDK:{m}) ✓ chars={len(t)} preview={_pv}…")
                            return t
                        print(f"[AI-LAWYER] Gemini (SDK:{m}) × Empty text.")
                    except Exception as e_try:
                        last_err = e_try
                        print(f"[AI-LAWYER] Gemini (SDK:{m}) × Exception: {e_try}")
                if last_err:
                    raise last_err
            except Exception as e_inner:
                print(f"[AI-LAWYER] Gemini (SDK) × All candidates failed: {e_inner}")
        except Exception as e2:
            print(f"[AI-LAWYER] Gemini (SDK) × Exception: {e2}")
        return None

    # Optional path: LangChain wrapper with memory
    try:
        chain = _maybe_init_lc_chain()
        if chain is not None:
            print("[AI-LAWYER] Gemini (LangChain Chain) → Generating IRAC answer …")
            resp = chain.run(prompt)
            if isinstance(resp, str) and resp.strip():
                text = resp.strip()
                _pv = text[:120].replace("\n", " ")
                print(f"[AI-LAWYER] Gemini (LangChain Chain) ✓ IRAC chars={len(text)} preview={_pv}…")
                return text
            print("[AI-LAWYER] Gemini (LangChain Chain) × Empty response.")
        else:
            print("[AI-LAWYER] Gemini (LangChain Chain) × Chain unavailable; falling back to SDK …")
    except Exception as e:
        print(f"[AI-LAWYER] Gemini (LangChain Chain) × Exception: {e}")
    # Fallback to SDK sequence when LC path didn't return text
    try:
        gen_cfg = {
            "temperature": 0.2,
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 2048,
        }
        print("[AI-LAWYER] Gemini (SDK Fallback) → Generating answer …")
        import google.generativeai as genai
        genai.configure(api_key=_get_gemini_api_key())
        candidates = [model_name, "gemini-2.0-flash-exp", "gemini-1.5-flash", "gemini-1.5-pro"]
        last_err = None
        for m in candidates:
            try:
                model = genai.GenerativeModel(m)
                resp = model.generate_content(prompt, generation_config=gen_cfg)
                text = getattr(resp, 'text', None)
                if text and text.strip():
                    t = text.strip()
                    _pv = t[:120].replace("\n", " ")
                    print(f"[AI-LAWYER] Gemini (SDK:{m}) ✓ chars={len(t)} preview={_pv}…")
                    return t
                print(f"[AI-LAWYER] Gemini (SDK:{m}) × Empty text.")
            except Exception as e_try:
                last_err = e_try
                print(f"[AI-LAWYER] Gemini (SDK:{m}) × Exception: {e_try}")
        if last_err:
            raise last_err
    except Exception as e2:
        print(f"[AI-LAWYER] Gemini (SDK Fallback) × Exception: {e2}")
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

    history_txt = _format_recent_history() or ""
    history_block = f"Conversation (recent):\n{history_txt}\n\n" if history_txt else ""
    prompt = f"""
{history_block}
You are assisting a legal chatbot. Generate a polished clarifying prompt using this exact format and tone.

Constraints:
- Do not show internal reasoning, scores, or tool names.
- Start with: "For a {role_key}, I can help you with: {caps_str}. A few quick details will help:"
    (Use an em dash — only if you naturally need one; do not include "Great — I’ll …".)
- Then list exactly three bullets using the • character, each ending with a question mark.
- Bullets should be:
    • Jurisdiction (India / UK / US / other)?
    • Year or exam (e.g., 2009–2023, semester finals)?
    • Topics you want to focus on (e.g., {examples}) — or type ‘any’ for a broad mix?

Context for you (not to be shown to the user):
- Role: {role or 'unknown'}
- Detected intent: {intent}
- Topic: {topic}
- Missing info hints: {mf_txt}
- User message: "{user_input}"

Return only the formatted text described above, nothing else.
"""
    model_name = _get_gemini_model_name()
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        llm = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=api_key,
            temperature=0.2,
            top_p=0.95,
            top_k=40,
            max_output_tokens=1024,
        )
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
            print("[AI-LAWYER] Gemini (SDK) → Generating clarifying questions …")
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            candidates = [model_name, "gemini-2.0-flash-exp", "gemini-1.5-flash", "gemini-1.5-pro"]
            last_err = None
            for m in candidates:
                try:
                    model = genai.GenerativeModel(m)
                    resp = model.generate_content(prompt)
                    text = getattr(resp, 'text', None)
                    if text and text.strip():
                        t = text.strip()
                        _pv = t[:120].replace("\n", " ")
                        print(f"[AI-LAWYER] Gemini (SDK:{m}) ✓ Clarify chars={len(t)} preview={_pv}…")
                        return t
                    print(f"[AI-LAWYER] Gemini (SDK:{m}) × Empty text.")
                except Exception as e_try:
                    last_err = e_try
                    print(f"[AI-LAWYER] Gemini (SDK:{m}) × Exception: {e_try}")
            if last_err:
                raise last_err
            return None
        except Exception as e2:
            print(f"[AI-LAWYER] Gemini (SDK) × All candidates failed: {e2}")
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

    # Silently learn stable preferences from this turn
    try:
        _merge_prefs(_extract_prefs_from_text(user_input))
    except Exception:
        pass

    # Role handling
    role_before = st.session_state.get("role")
    if role_before is None:
        # If user already told us their role earlier, use it silently
        mem_role = _prefs().get("role")
        if isinstance(mem_role, str) and mem_role in {"student", "professional"}:
            st.session_state["role"] = mem_role
        else:
            detected = understanding.classify_role(user_input)
            if detected:
                st.session_state["role"] = detected
                # Present a quick role intro once, then continue
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
    # Learn additional prefs (topic/role) from analysis silently
    try:
        _update_prefs_from_analysis(analysis)
    except Exception:
        pass

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
            # Avoid asking for things already in memory
            missing_fields: List[str] = analysis.get("missing_fields", []) or []
            mf = []
            p = _prefs()
            for f in missing_fields:
                fl = f.lower()
                if ("jurisdiction" in fl or "court" in fl) and p.get("jurisdiction"):
                    continue
                if ("timeframe" in fl or "year" in fl or "date" in fl) and p.get("exam_years"):
                    continue
                if ("topic" in fl) and p.get("topics"):
                    continue
                mf.append(f)

            if len(mf) == 0:
                # Proceed using stored preferences silently
                assumed = {
                    "jurisdiction": p.get("jurisdiction") or "India",
                    "years": p.get("exam_years") or "recent 5 years",
                    "topics": p.get("topics") or (
                        analysis.get("topic") if (analysis.get("topic") and analysis.get("topic") != "general") else "core relevant topics"
                    ),
                }
            else:
                clarify = _llm_clarify(
                    user_input,
                    role,
                    analysis.get("intent", "legal_query"),
                    analysis.get("topic", "general"),
                    mf,
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
    # Apply style preference to the user query for downstream tools
    styled_query = _apply_style_to_query(user_input, analysis.get("intent", "legal_query"))

    retrieved = rag.search_chunks(
        styled_query,
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
        enriched_query = styled_query + assumed_text
        answer = _llm_irac_answer(enriched_query, role, intent, topic, retrieved)
    else:
        answer = _llm_irac_answer(styled_query, role, intent, topic, retrieved)
    if not answer:
        answer = (
            "I'm currently unavailable to proceed due to an LLM request failure. "
            "Please verify your internet/API quota and try a supported model (e.g., gemini-1.5-flash). "
            "You can also set USE_LANGCHAIN_GENAI=0 in .env to use the SDK path."
        )

    with st.chat_message("assistant"):
        st.markdown(answer)
    st.session_state["messages"].append({"role": "assistant", "content": answer})
    st.session_state["last_assistant_type"] = "final"


if __name__ == "__main__":
    main()
