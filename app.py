from __future__ import annotations
import os
import re
import streamlit as st
from typing import Dict, List, Optional, Any
from pathlib import Path
from dotenv import load_dotenv

# Prefer absolute imports
import understanding
import rag
import student_tools
import professional_tools

# ---------------------------------------------------------
# CONFIG & SETUP
# ---------------------------------------------------------
APP_TITLE = "LexiCounsel — AI Lawyer MVP"

try:
    load_dotenv(dotenv_path=Path(__file__).with_name(".env"), override=False)
except Exception:
    pass

# ---------------------------------------------------------
# HELPER: CLI LOGGING
# ---------------------------------------------------------
def log_cli(section: str, message: str):
    print(f"[{section}] {message}")

# ---------------------------------------------------------
# SESSION & MEMORY MANAGEMENT
# ---------------------------------------------------------
def _init_session():
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    
    if "role" not in st.session_state:
        st.session_state["role"] = None 
    
    if "user_prefs" not in st.session_state:
        st.session_state["user_prefs"] = {
            "role": None,
            "jurisdiction": "India", 
            "style": None,           
            "topics": None,          
            "exam_years": None       
        }
    
    if "hitl_active" not in st.session_state:
        st.session_state["hitl_active"] = False

def _update_memory(new_prefs: Dict[str, str]):
    if not new_prefs: return
    changes = []
    for k, v in new_prefs.items():
        if v and st.session_state["user_prefs"].get(k) != v:
            st.session_state["user_prefs"][k] = v
            changes.append(f"{k}={v}")
    if changes:
        log_cli("🧠 Memory", f"Updated: {', '.join(changes)}")

def _extract_prefs_from_text(text: str) -> Dict[str, str]:
    t = text.lower()
    prefs = {}
    if re.search(r"\b(india|indian)\b", t): prefs["jurisdiction"] = "India"
    elif re.search(r"\b(uk|united kingdom|english)\b", t): prefs["jurisdiction"] = "UK"
    elif re.search(r"\b(us|usa|american)\b", t): prefs["jurisdiction"] = "US"

    if "no irac" in t or "avoid irac" in t: prefs["style"] = "Natural Summary (No IRAC)"
    elif "use irac" in t or "prefer irac" in t: prefs["style"] = "Strict IRAC"
    elif "bullet" in t: prefs["style"] = "Bullet Points"
    elif "simplify" in t: prefs["style"] = "Simple English (EL15)"

    years = re.findall(r"\b(20\d{2})\b", t)
    if years: prefs["exam_years"] = ", ".join(sorted(set(years)))

    if "don't store" in t or "temporary" in t: return {}
    return prefs

# ---------------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------------

def _get_gemini_api_key() -> Optional[str]:
    key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if key: return key
    try:
        return st.secrets.get("GEMINI_API_KEY")
    except:
        return None

def _get_gemini_model_name() -> str:
    return os.getenv("GEMINI_GEN_MODEL", "gemini-2.5-flash")

def _is_trivial_query(text: str) -> bool:
    t = text.strip().lower()
    trivial_phrases = {
        "nothing", "no", "nope", "na", "ok", "okay", "thanks", "thank you", 
        "thx", "cool", "hello", "hi", "hey", "greetings", "bye", "goodbye",
        "good morning", "yo", "done"
    }
    if t in trivial_phrases: return True
    if len(t) < 4 and t not in ["ipc", "crpc", "cpc", "fir"]: return True
    return False

# --- NEW: Short-Term Memory Helper ---
def _format_recent_history(max_turns: int = 4) -> str:
    """Formats the last few messages to give the LLM context."""
    msgs = st.session_state.get("messages", [])
    if not msgs: return ""
    
    # Get last N messages (excluding the very current user input which is handled separately)
    history = msgs[-max_turns:] 
    formatted = []
    for m in history:
        role = "User" if m["role"] == "user" else "Assistant"
        formatted.append(f"{role}: {m['content']}")
    
    return "\n".join(formatted)

# ---------------------------------------------------------
# UI COMPONENTS
# ---------------------------------------------------------

@st.dialog("Welcome to LexiCounsel ⚖️")
def role_selection_popup():
    st.write("To customize your legal assistant, please select your profile:")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🎓 I am a Law Student", use_container_width=True):
            st.session_state["role"] = "student"
            st.session_state["user_prefs"]["role"] = "student"
            st.rerun()
    with col2:
        if st.button("💼 I am a Professional", use_container_width=True):
            st.session_state["role"] = "professional"
            st.session_state["user_prefs"]["role"] = "professional"
            st.rerun()

def _send_intro_message():
    role = st.session_state["role"]
    if not role: return
    jurisdiction = st.session_state["user_prefs"].get("jurisdiction", "India")
    
    if role == "student":
        msg = (
            f"**Hello! I'm LexiCounsel, your AI Study Companion ({jurisdiction}).** 🎓\n\n"
            "I can help you with **PYQs**, **Case Summaries**, and **Mock Arguments**.\n"
            "What are you studying today?"
        )
    else:
        msg = (
            f"**Greetings. I'm LexiCounsel, your Legal Research Assistant ({jurisdiction}).** ⚖️\n\n"
            "I can assist with **Case Law**, **Drafting**, and **Citations**.\n"
            "How can I assist with your practice today?"
        )
            
    if not st.session_state["messages"]:
        st.session_state["messages"].append({"role": "assistant", "content": msg})

# ---------------------------------------------------------
# LLM GENERATION (MEMORY + CONTEXT AWARE)
# ---------------------------------------------------------

def _generate_answer(query: str, role: str, intent: str, retrieved_docs: List[Dict]) -> str:
    log_cli("🤖 LLM", "Generating final answer with HISTORY...")
    api_key = _get_gemini_api_key()
    if not api_key: return "⚠️ Error: API Key missing."

    # 1. User Preferences (Long Term)
    prefs = st.session_state["user_prefs"]
    memory_block = f"""
    [USER PREFERENCES]
    - Role: {prefs.get('role', role)}
    - Jurisdiction: {prefs.get('jurisdiction', 'India')}
    - Preferred Style: {prefs.get('style', 'Default')}
    """

    # 2. Conversation History (Short Term) - CRITICAL FIX
    # We grab previous turns so the LLM knows what "that case" refers to.
    history_text = _format_recent_history(max_turns=6)
    
    # 3. Retrieved Docs
    context_text = ""
    for idx, doc in enumerate(retrieved_docs):
        yr = doc.get('year')
        pg = doc.get('page')
        summary = doc.get('summary') or doc.get('text') or ""
        # Avoid exposing filenames or PDF names in model context
        context_text += f"\n[Case {idx+1}] Year: {yr}, Page: {pg}\n{summary}\n"
    has_context = len(retrieved_docs) > 0

    # 4. Instructions
    default_instr = "Style: IRAC." if role == "student" else "Style: Professional."
    override_instr = ""
    if not has_context:
        # Strict guardrail: avoid conclusions when corpus provides no match
        override_instr = (
            "No specific matches found in the available corpus. "
            "Do not infer outcomes or procedural steps not present in the retrieved context. "
            "If the user asks about post-judgment appeals or subsequent litigation, respond that the provided corpus does not mention such details, "
            "and suggest narrowing timeframe, providing a citation, or uploading the relevant documents."
        )

    prompt = f"""
    You are LexiCounsel, an expert AI Lawyer.
    
    {memory_block}

    [CONVERSATION HISTORY]
    {history_text}
    
    [RETRIEVED CONTEXT (If any specific to current query)]
    {context_text if context_text else "(No specific database matches found for this specific turn)"}
    
    [CURRENT QUERY]
    User: "{query}"
    Intent: {intent}
    
    [INSTRUCTIONS]
    1. **Context Awareness**: Use [CONVERSATION HISTORY] to understand references like "explain that case" or "what did you mean".
    2. **Adopt Jurisdiction**: {prefs.get('jurisdiction')}.
    3. **Apply Style**: {prefs.get('style')} or {default_instr}.
    4. {override_instr}
    5. Do not reveal PDF names, filenames, paths, or internal source identifiers unless the user explicitly asks for sources.
    6. If the retrieved context is empty or not clearly relevant, avoid speculating; ask for clarification or narrower timeframe.
    
    [RESPONSE]
    """

    # If there is no context, prefer a cautious, direct response to prevent hallucinations
    if not has_context:
        cautious = (
            "Based on the currently available corpus, there is no mention of post-judgment appeals or Tribunal proceedings by Ambuja. "
            "If you can share the specific citation or timeframe, I can check more precisely."
        )
        log_cli("🛡️ Guardrail", "Returned cautious response due to empty retrieval context.")
        return cautious

    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(_get_gemini_model_name())
        resp = model.generate_content(prompt)
        return resp.text
    except Exception as e:
        return f"⚠️ Error: {e}"

# ---------------------------------------------------------
# MAIN APP LOOP
# ---------------------------------------------------------

def main():
    st.set_page_config(page_title=APP_TITLE, page_icon="⚖️", layout="centered")
    _init_session()

    if st.session_state["role"] is None:
        role_selection_popup()
    else:
        if not st.session_state["messages"]:
            _send_intro_message()

    for m in st.session_state["messages"]:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    user_input = st.chat_input("Type your legal query...")
    if not user_input: return

    print("-" * 50)
    log_cli("👤 User Input", user_input)
    
    # Add to history *before* processing so _format_recent_history sees it? 
    # Actually, usually better to let the prompt see history *up to* current, 
    # and pass current as "User Query".
    # So we display it first:
    st.session_state["messages"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Silent Memory Update
    new_prefs = _extract_prefs_from_text(user_input)
    if new_prefs: _update_memory(new_prefs)

    # Chitchat
    if _is_trivial_query(user_input):
        log_cli("🧠 Logic", "Trivial query.")
        # Basic context-aware chitchat
        # If user says "thanks", look at history to see what they are thanking for? 
        # For now, keep it simple.
        response = "You're welcome! Let me know if you need anything else."
        st.session_state["messages"].append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.markdown(response)
        return

    # HITL Check
    if st.session_state["hitl_active"]:
        st.session_state["hitl_active"] = False 

    # Understanding
    role = st.session_state["role"]
    analysis = understanding.analyze_query(user_input, role)
    
    # Guardrail
    confidence = (float(analysis.get("intent_confidence", 0)) + float(analysis.get("topic_confidence", 0))) / 2.0
    if confidence < 0.5:
        # Check history length - if we are deep in convo, maybe low confidence is fine 
        # (user might be saying "explain more")
        if len(st.session_state["messages"]) > 4:
            log_cli("⚠️ Guardrail", "Low confidence but allowing due to conversation history.")
        else:
            log_cli("⚠️ Guardrail", "Confidence low. Triggering HITL.")
            st.session_state["hitl_active"] = True
            fallback_msg = f"I'm not sure I understood. Did you mean to ask about **{analysis.get('topic')}**?"
            st.session_state["messages"].append({"role": "assistant", "content": fallback_msg})
            with st.chat_message("assistant"):
                st.markdown(fallback_msg)
            return

    # RAG
    with st.spinner("Analyzing..."):
        court_filter = None
        if "supreme" in user_input.lower(): court_filter = "scc"
        elif "high" in user_input.lower(): court_filter = "hcc"
        import re
        year_match = re.search(r"\b(19|20)\d{2}\b", user_input)
        year_min = int(year_match.group(0)) if year_match else None

        retrieved_docs = rag.search_chunks(
            query=user_input,
            top_k=5,
            court=court_filter,
            year_min=year_min
        )

        # --- CLI transparency: log filters and retrieved docs ---
        log_cli("🔎 Filters", f"court={court_filter} year_min={year_min}")
        if not retrieved_docs:
            log_cli("📚 Retrieval", "No documents matched from corpus (2005–2006 PDFs).")
        else:
            for i, d in enumerate(retrieved_docs, start=1):
                log_cli(
                    "📚 Retrieval",
                    f"{i}. file={d.get('doc_id')} year={d.get('year')} page={d.get('page')} score={round(d.get('score', 0), 3)} cat={d.get('category')}"
                )
        # --- Guardrail: corpus-limited disclaimer when weak/no context ---
        disclaimer = None
        # Known corpus window based on your ingestion
        corpus_min, corpus_max = 2005, 2006
        query_year = year_min
        if not retrieved_docs:
            disclaimer = (
                "⚠️ No matching judgments found in the current corpus (2005–2006 PDFs). "
                "If you’re asking about events outside this range (e.g., 2G spectrum in 2010–2012), "
                "please provide a timeframe within 2005–2006 or upload the relevant years."
            )
        elif query_year and (query_year < corpus_min or query_year > corpus_max):
            disclaimer = (
                f"⚠️ Your query mentions {query_year}, but the available corpus spans {corpus_min}–{corpus_max}. "
                "Results may be incomplete; consider specifying cases or years within the corpus."
            )

        answer = _generate_answer(user_input, role, analysis["intent"], retrieved_docs)

    st.session_state["messages"].append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        # Show optional disclaimer first
        if disclaimer:
            st.markdown(disclaimer)
            st.divider()
        st.markdown(answer)
        # Sources panel
        if retrieved_docs:
            with st.expander("Sources (retrieved)"):
                for i, d in enumerate(retrieved_docs, start=1):
                    title = d.get("title") or d.get("doc_id")
                    meta = f"Year: {d.get('year')}"
                    st.markdown(f"- **{i}. {title}** — {meta}")

if __name__ == "__main__":
    main()