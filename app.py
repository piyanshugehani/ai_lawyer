from __future__ import annotations
import os
import re
import json
import time
import streamlit as st
from typing import Dict, List, Optional, Any
from pathlib import Path
from dotenv import load_dotenv

# Prefer absolute imports
import understanding
import rag
import student_tools
import professional_tools
import arguments_panel

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

    # Tracks whether the professional user is in an ongoing drafting flow
    if "drafting_active" not in st.session_state:
        st.session_state["drafting_active"] = False

    # Whether to also include High Court (HCC) cases along with Supreme Court (SCC)
    if "include_high_court" not in st.session_state:
        st.session_state["include_high_court"] = False

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

# --- Short-Term Memory Helper ---
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

    # 2. Conversation History (Short Term)
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
    5. Prefer High Court judgments for procedural or registry-related questions (e.g. dismissal for default, office objections) when such context is available.
     6. If case number, court, or year mismatch exists between the user's query and the retrieved context, you MUST refuse to answer and MUST NOT generalize from other cases; instead respond that the retrieved context does not match the specified case details.
     7. Do NOT merge or conflate multiple different cases in a single answer when the user has specified a particular case identity.
     8. Do not reveal PDF names, filenames, paths, or internal source identifiers unless the user explicitly asks for sources.
     9. If the retrieved context is empty or not clearly relevant, avoid speculating; ask for clarification or a narrower timeframe.
    
    [RESPONSE]
    """

    # If there is no context, prefer a cautious, direct response to prevent hallucinations
    if not has_context:
        cautious = (
            "Based on the currently available corpus, there is no mention of such details. "
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


def _generate_legal_draft(query: str, history_text: str = "") -> str:
    """Generate a court-ready Indian legal draft for professionals.

    Follows "LexCrest Legal Drafting Assistant" behaviour:
    - Plain text only (no markdown/JSON)
    - May ask clarifying questions first if essentials missing
    - Appends mandatory disclaimer line at the end
    """
    log_cli("📝 Draft", "Generating legal draft for professional user…")
    api_key = _get_gemini_api_key()
    if not api_key:
        return "Error: LLM API key is missing. Configure GEMINI_API_KEY or GOOGLE_API_KEY before using drafting."

    prefs = st.session_state.get("user_prefs", {})
    jurisdiction = prefs.get("jurisdiction", "India")

    prompt = f"""
You are LexCrest Legal Drafting Assistant, an AI specialized in Indian legal drafting.

The user interacts via a normal chat interface and may simply say things like:
- "Create a legal notice for non-payment of rent"
- "Draft an affidavit for name change"

Your task is to generate a COMPLETE, COURT-READY legal draft for Indian jurisdiction (default: {jurisdiction}).

[CONVERSATION HISTORY]
{history_text}

[CURRENT USER INPUT]
{query}

--------------------------------
IMPORTANT OUTPUT RULE
--------------------------------
Your response will be directly converted into a PDF in Streamlit.

Therefore:
- Output ONLY the legal draft text
- Do NOT include explanations, greetings, or summaries
- Do NOT use markdown (no *, #, -, bullets)
- Do NOT return JSON
- Do NOT say phrases like "Here is the draft" or similar wrappers
- Do NOT include placeholders like [Signature] unless it is part of the form.

Return plain text only, suitable for direct PDF rendering.

--------------------------------
CLARIFICATION RULE
--------------------------------
If essential legal details are missing for a valid draft (such as names of parties, dates, property description, amount in dispute, or relevant court):
1. **Check History**: Look at [CONVERSATION HISTORY]. If the user just answered a question about these details, USE them.
2. **Ask Questions**: Only if details are STILL missing, ask the minimal required questions in ONE single message.
3. **Generate**: If all essential details are present (either in current input or history), do NOT ask questions; generate the draft directly.

--------------------------------
DRAFTING RULES
--------------------------------
- Use formal Indian legal language
- Use professional advocate-style tone
- Proper structure is mandatory:

1. DOCUMENT TITLE (centered text style)
2. PARTIES
3. FACTS / BACKGROUND
4. LEGAL PROVISIONS (if applicable)
5. CLAUSES / RELIEF SOUGHT (numbered)
6. JURISDICTION
7. SIGNATURE BLOCK

- Number clauses properly (1., 1.1, 1.2 if needed)
- Use line breaks for readability
- Assume A4, justified layout (plain text with suitable line breaks)
- No emojis, no decorative symbols

--------------------------------
LEGAL DISCLAIMER (MANDATORY)
--------------------------------
Append this EXACT text as the final line of the draft:

This document is AI-generated and must be reviewed by a qualified legal professional before use.
"""

    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(_get_gemini_model_name())
        resp = model.generate_content(prompt)
        return resp.text
    except Exception as e:
        return f"Error while generating draft: {e}"


# (Arguments panel / moot-court helpers now live in arguments_panel.py)

# ---------------------------------------------------------
# MAIN APP LOOP
# ---------------------------------------------------------

def main():
    st.set_page_config(page_title=APP_TITLE, page_icon="⚖️", layout="centered")
    _init_session()

    # Sidebar controls for retrieval behaviour
    with st.sidebar:
        st.markdown("### Corpus Settings")
        st.checkbox(
            "Include High Court cases (with Supreme Court)",
            key="include_high_court",
            help=(
                "When enabled, retrieval will search both Supreme Court (SCC) and "
                "High Court (HCC) judgments. When disabled, queries default to "
                "Supreme Court judgments unless you explicitly mention High Court."
            ),
        )

    if st.session_state["role"] is None:
        role_selection_popup()
        return

    # Shared intro + history rendering
    if not st.session_state["messages"]:
        _send_intro_message()

    role = st.session_state["role"]

    # For professionals, expose an additional "Arguments" area alongside chat.
    if role == "professional":
        tab_chat, tab_arguments = st.tabs(["Chat", "Arguments"])
    else:
        # For students, keep a single chat surface.
        tab_chat, tab_arguments = st.container(), None

    with tab_chat:
        for m in st.session_state["messages"]:
            with st.chat_message(m["role"]):
                st.markdown(m["content"])

        user_input = st.chat_input("Type your legal query...")
        if not user_input:
            # Still render Arguments tab for professionals even if no new chat input
            if role == "professional" and tab_arguments is not None:
                with tab_arguments:
                    arguments_panel.render_arguments_panel()
            return

        print("-" * 50)
        log_cli("👤 User Input", user_input)
        
        st.session_state["messages"].append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

    # Silent Memory Update
    new_prefs = _extract_prefs_from_text(user_input)
    if new_prefs: _update_memory(new_prefs)

    # Chitchat
    if _is_trivial_query(user_input):
        log_cli("🧠 Logic", "Trivial query.")
        response = "You're welcome! Let me know if you need anything else."
        st.session_state["messages"].append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.markdown(response)
        return

    # HITL Check
    if st.session_state["hitl_active"]:
        st.session_state["hitl_active"] = False 

    # Understanding
    analysis = understanding.analyze_query(user_input, role)
    intent = analysis.get("intent", "legal_query")
    
    # If professional and drafting intent (or already in drafting flow), use dedicated drafting assistant path
    if role == "professional" and (intent == "draft_notice" or st.session_state.get("drafting_active")):
        # Mark drafting as active so follow-up detail messages also go through this path
        st.session_state["drafting_active"] = True
        log_cli("📝 Route", "Professional drafting flow active using LexCrest drafting assistant.")
        
        # 1. GET HISTORY (Fixes Amnesia)
        history_text = _format_recent_history(max_turns=10)

        # 2. GENERATE DRAFT WITH HISTORY
        draft_text = _generate_legal_draft(user_input, history_text=history_text)
        
        st.session_state["messages"].append({"role": "assistant", "content": draft_text})
        with st.chat_message("assistant"):
            # Show raw text (no markdown) to respect plain-text requirement
            st.text(draft_text)

            # --- PDF GENERATION & DOWNLOAD ---
            try:
                from fpdf import FPDF
                
                class PDF(FPDF):
                    def header(self):
                        # Use Helvetica (core font) to avoid Arial warnings
                        self.set_font('Helvetica', 'B', 14)
                        # 'align' keyword is safer for newer fpdf2 versions
                        self.cell(0, 10, 'LEGAL DRAFT (AI GENERATED)', align='C', new_x="LMARGIN", new_y="NEXT")
                        self.ln(5)
                
                # Init PDF
                pdf = PDF()
                pdf.add_page()
                pdf.set_font("Helvetica", size=11)

                # Process text to fix encoding
                for raw_line in draft_text.splitlines():
                    # 1. Sanitize text: Replace unsupported chars
                    safe_line = raw_line.replace("₹", "Rs. ").replace("–", "-").replace("“", '"').replace("”", '"')
                    
                    # 2. Encode/Decode to ensure Latin-1 compatibility (Core fonts don't support Unicode)
                    safe_line = safe_line.encode("latin-1", "replace").decode("latin-1")
                    
                    if not safe_line.strip():
                        pdf.ln(5)
                        continue
                    
                    # 3. Write line
                    # w=0 means "use all available width". 
                    # We explicitely set x to left margin to prevent "Not enough horizontal space" errors
                    pdf.set_x(pdf.l_margin) 
                    pdf.multi_cell(w=0, h=5, text=safe_line)

                # 4. Create Bytes for Download Button
                # fpdf2's output() returns a bytearray. We cast it to bytes for Streamlit.
                pdf_bytes = bytes(pdf.output())
                
                st.download_button(
                    label="Download Draft as PDF",
                    data=pdf_bytes,
                    file_name="legal_draft.pdf",
                    mime="application/pdf",
                )
            except Exception as e:
                st.error(f"Failed to build PDF: {e}")
                log_cli("PDF", f"Error: {e}")
        return

    # Guardrail for non-drafting flows
    intent_conf = float(analysis.get("intent_confidence", 0))
    topic_conf = float(analysis.get("topic_confidence", 0))
    confidence = (intent_conf + topic_conf) / 2.0

    log_cli(
        "🧩 Understanding",
        f"intent={analysis.get('intent')} topic={analysis.get('topic')} "
        f"intent_conf={intent_conf:.2f} topic_conf={topic_conf:.2f} combined={confidence:.2f}",
    )
    if confidence < 0.5:
        # Check history length - if we are deep in convo, maybe low confidence is fine 
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
        include_hcc = st.session_state.get("include_high_court", False)

        # Parse case identity from the user query (case number, year, court)
        case_identity = understanding.parse_case_identity(user_input)

        # Heuristic: treat certain keywords as procedural / registry issues
        lower_q = user_input.lower()

        # Detect if this is explicitly a High Court query (by text or identity)
        is_high_court_query = (
            "high court" in lower_q or case_identity.get("court_level") == "High Court"
        )

        # Court routing logic (matches rag.search_chunks behaviour):
        # - When toggle is OFF  -> strictly Supreme Court only ("scc"), even if
        #   the query text mentions a High Court.
        # - When toggle is ON   ->
        #       * if query clearly targets a High Court -> High Court index only ("hc")
        #       * otherwise -> both Supreme Court and High Court ("both").
        court_filter: Optional[str]
        if not include_hcc:
            court_filter = "scc"
        else:
            if is_high_court_query:
                court_filter = "hc"
            else:
                court_filter = "both"
        procedural_keywords = [
            "dismissed for default",
            "dismissed for non removal of office objection",
            "non removal of office objection",
            "office objection",
            "restoration application",
            "delay condonation",
            "condonation of delay",
            "restored to file",
        ]
        is_procedural = any(k in lower_q for k in procedural_keywords)

        import re
        # Year constraints:
        # - For queries that hit only the Supreme Court index (court_filter == "scc"),
        #   you may continue using filing_year / decision_year style constraints
        #   (via filing_year / first year in query).
        # - For queries that involve the High Court corpus (court_filter != "scc"),
        #   do NOT restrict by filing_year/decision_year so that slight metadata
        #   differences in PDFs do not cause false negatives.
        filing_year = case_identity.get("filing_year")
        year_min: Optional[int]
        year_max: Optional[int]
        if court_filter == "scc":
            if filing_year and filing_year.isdigit():
                year_min = year_max = int(filing_year)
            else:
                year_match = re.search(r"\b(19|20)\d{2}\b", user_input)
                year_min = int(year_match.group(0)) if year_match else None
                year_max = None
        else:
            year_min = None
            year_max = None

        # When High Court corpus is enabled, do not apply any extra "procedural"
        # boosting so that bail / rejection / dismissal details are not filtered out.
        # We simply let the retriever rank by semantic similarity.
        prefer_high_court = False

        retrieved_docs = rag.search_chunks(
            query=user_input,
            top_k=5,
            court=court_filter,
            year_min=year_min,
            year_max=year_max,
            prefer_high_court=prefer_high_court,
        )

        # --- CLI transparency: log filters and retrieved docs ---
        log_cli(
            "🔎 Filters",
            f"court={court_filter} year_min={year_min} include_high_court={include_hcc} "
            f"procedural={is_procedural} case_identity={case_identity}",
        )
        if not retrieved_docs:
            log_cli("📚 Retrieval", "No documents matched from corpus (2005–2006 PDFs).")
        else:
            for i, d in enumerate(retrieved_docs, start=1):
                log_cli(
                    "📚 Retrieval",
                    f"{i}. file={d.get('doc_id')} year={d.get('year')} page={d.get('page')} "
                    f"score={round(d.get('score', 0), 3)} cat={d.get('category')}"
                )
        # --- Guardrail: corpus-limited disclaimer when weak/no context ---
        disclaimer = None
        # For precise identity queries (explicit case_number + filing_year),
        # skip generic corpus-year warnings to avoid confusing filing vs
        # decision years.
        is_identity_query = bool(case_identity.get("case_number") and case_identity.get("filing_year"))
        if not is_identity_query:
            # Known corpus window based on your ingestion of PDFs
            corpus_min, corpus_max = 2005, 2006
            query_year = year_min
            if not retrieved_docs:
                disclaimer = (
                    "⚠️ No matching judgments found in the current uploaded corpus. "
                    "If you’re asking about events clearly outside the years covered "
                    "by this corpus, please provide a narrower timeframe or upload "
                    "the relevant documents."
                )
            elif query_year and (query_year < corpus_min or query_year > corpus_max):
                disclaimer = (
                    "⚠️ Your query mentions a year that may fall outside the primary "
                    "coverage of the current corpus. Results may be incomplete; "
                    "consider specifying cases or years known to be within the corpus."
                )

        # --- Identity check: if user specified a case identity, ensure retrieved
        # context actually matches it; otherwise, abort with safe message.
        def _doc_matches_identity(doc: Dict[str, Any], ident: Dict[str, Any]) -> bool:
            if not ident:
                return True

            text_block = (doc.get("text") or doc.get("summary") or "").lower()
            fname = (doc.get("doc_id") or "").lower()

            # 1) Exact header phrase match wins immediately
            raw = (ident.get("raw_string") or "").lower()
            if raw and raw in text_block:
                return True

            num = ident.get("case_number")
            filing_year = ident.get("filing_year")
            court_name = (ident.get("court_name") or "").lower()

            # Case number check (mandatory if specified)
            if num:
                # Check if number exists in text or filename
                if num not in text_block and num not in fname:
                    return False

            # Filing year check
            if filing_year:
                # ROBUSTNESS FIX: Check both 'court' and 'category' for HC indicator
                doc_court = (doc.get("court") or "").lower()
                doc_cat = (doc.get("category") or "").lower()
                
                is_high_court_doc = "high court" in doc_court or "high court" in doc_cat

                # If it is NOT a High Court doc, we generally expect strict year matching.
                # However, we add a buffer because Filing Year (User) <= Decision Year (Doc)
                if not is_high_court_doc:
                    try:
                        doc_year = int(doc.get("year") or 0)
                        req_year = int(filing_year)
                        
                        # Allow the document to be the same year OR up to 3 years later
                        # (e.g. Case filed in 2004, decided in 2005)
                        if not (req_year <= doc_year <= req_year + 3):
                            return False
                    except ValueError:
                        # If year parsing fails, be permissive to avoid false negatives
                        pass

            # Court name check
            if court_name:
                doc_court = (doc.get("court") or "").lower()
                doc_cat = (doc.get("category") or "").lower()
                
                # If user explicitly wants "Supreme Court", ensure doc isn't "High Court"
                if "supreme" in court_name and ("high court" in doc_court or "high court" in doc_cat):
                    return False

            return True
        # If we have an explicit case identity (number or filing year), enforce
        # a hard match; otherwise fall back to normal behaviour.
        if case_identity.get("case_number") or case_identity.get("filing_year"):
            filtered_docs = [d for d in retrieved_docs if _doc_matches_identity(d, case_identity)]
            if not filtered_docs:
                answer = "The retrieved context does not match the specified case details."
                retrieved_docs = []  # Avoid leaking mismatched context downstream
            else:
                retrieved_docs = filtered_docs
                answer = _generate_answer(user_input, role, analysis["intent"], retrieved_docs)
        else:
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

    # After handling chat, also render Arguments panel for professional users
    if role == "professional" and tab_arguments is not None:
        with tab_arguments:
            arguments_panel.render_arguments_panel()


if __name__ == "__main__":
    main()