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

    # Embedding model configuration for RAG (controls rag.EMBEDDING_MODEL)
    if "embedding_model" not in st.session_state:
        # Default to whatever the rag module is currently using
        try:
            st.session_state["embedding_model"] = getattr(rag, "EMBEDDING_MODEL", "models/gemini-embedding-001")
        except Exception:
            st.session_state["embedding_model"] = "models/gemini-embedding-001"

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

def _inject_global_styles() -> None:
    """Inject a dark-mode-friendly, professional visual theme."""
    st.markdown(
        """
        <style>
            /* Constrain main width for a more app-like feel */
            .block-container {
                max-width: 1100px;
                padding-top: 1.0rem;
                padding-bottom: 1.8rem;
            }

            /* Top header branding – neutral so it works on dark/light */
            .lc-header {
                padding-bottom: 0.7rem;
                margin-bottom: 0.5rem;
                border-bottom: 1px solid rgba(255, 255, 255, 0.12);
            }
            .lc-header-title {
                font-size: 1.6rem;
                font-weight: 700;
                margin-bottom: 0.1rem;
            }
            .lc-header-subtitle {
                font-size: 0.9rem;
                opacity: 0.75;
                margin: 0;
            }

            /* Compact status badges under the header */
            .lc-status-row {
                display: flex;
                flex-wrap: wrap;
                gap: 0.4rem;
                margin-top: 0.25rem;
                margin-bottom: 0.4rem;
            }
            .lc-badge {
                font-size: 0.75rem;
                padding: 0.12rem 0.55rem;
                border-radius: 999px;
                background: rgba(255, 255, 255, 0.06);
                border: 1px solid rgba(255, 255, 255, 0.12);
                backdrop-filter: blur(8px);
            }

            /* Chat message bubbles – tuned for dark mode */
            div[data-testid="stChatMessage"] {
                border-radius: 10px;
                padding: 0.35rem 0.55rem;
                margin-bottom: 0.35rem;
                background: rgba(255, 255, 255, 0.02);
                border: 1px solid rgba(255, 255, 255, 0.07);
            }

            /* Slight differentiation: last user vs assistant bubble alignment */
            div[data-testid="stChatMessage"]:has(div[data-testid="stMarkdownContainer"] p:contains("👤 User")) {
                text-align: left;
            }

            /* Footer */
            .lc-footer {
                font-size: 0.75rem;
                opacity: 0.7;
                border-top: 1px solid rgba(255, 255, 255, 0.12);
                margin-top: 1.0rem;
                padding-top: 0.55rem;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

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

@st.dialog("Welcome to LexiCounsel (India) ⚖️")
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
            "I can help you with **PYQs**, **Case Summaries**,**Section Simplified** and **Bare acts**.\n"
            "What are you studying today?"
        )
    else:
        msg = (
            f"**Greetings. I'm LexiCounsel, your Legal Research Assistant ({jurisdiction}).** ⚖️\n\n"
            "I can assist with **Case Law**, **Drafting**, **Mock Arguments** and **Citations**.\n"
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
        override_instr = (
            "No specific judgments from the internal corpus are available for this turn. "
            "You may still answer using your general legal knowledge for the relevant jurisdiction, "
            "but clearly state that the answer is based on general principles and not on a specific retrieved case. "
            "Avoid fabricating case names, citations, or procedural histories. If the user needs authority, "
            "invite them to provide a citation, timeframe, or upload the relevant documents."
        )
    else:
        # When we DO have retrieved context, explicitly tell the model to
        # ground its answer in that context where possible, but still allow
        # fallback to its wider legal knowledge for well-known cases.
        override_instr = (
            "There IS retrieved context available for this turn. Base your answer "
            "primarily on the [RETRIEVED CONTEXT] text. Do NOT say that there is "
            "no specific retrieved document or that you cannot answer from the "
            "retrieved documents. If the context is incomplete for the exact "
            "question, still provide the best possible answer that is faithful "
            "to the retrieved text, and mention that the corpus snippet may be partial. "
            "If the user asks about a well-known case (by name and year) and the "
            "retrieved snippets do not clearly contain that case, you may also use "
            "your general legal knowledge of that case, clearly labelling those parts "
            "as based on general legal understanding beyond the uploaded corpus."
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
     6. Only when the user has provided a very specific case identity (such as an exact case number or neutral citation) and the retrieved context clearly does not match that identity, you MUST refuse to answer from the corpus and say that the retrieved context does not match those details. In all other situations, you may answer using a combination of retrieved snippets and your general legal knowledge, with clear labelling.
     7. Do NOT merge or conflate multiple different cases in a single answer when the user has specified a particular case identity.
     8. Do not reveal PDF names, filenames, paths, or internal source identifiers unless the user explicitly asks for sources.
     9. If the retrieved context is empty or clearly off-topic, avoid speculating; ask for clarification or a narrower timeframe. If there IS retrieved context, answer from it as faithfully as possible even if it does not spell out every detail of the user's question, and where necessary supplement with clearly-labelled general legal knowledge.
    
    [RESPONSE]
    """

    try:
        from google import genai
        client = genai.Client(api_key=api_key)
        model_name = _get_gemini_model_name()
        resp = client.models.generate_content(model=model_name, contents=prompt)
        # New google-genai client returns a response object with 'text' attr
        return getattr(resp, "text", str(resp))
    except Exception as e:
        return f"⚠️ Error: {e}"


def _format_citation_block(retrieved_docs: List[Dict[str, Any]]) -> str:
    """Build a markdown citations block from retrieved documents.

    This is appended AFTER the main answer so that the LLM
    doesn't have to guess sources; we use the actual retrieved
    context instead. Shown only when we have at least one doc.
    """
    if not retrieved_docs:
        return ""

    # lines: List[str] = []
    # lines.append("\n\n---\n**Sources / Citations (retrieved)**\n")

    # for idx, doc in enumerate(retrieved_docs, start=1):
    #     # Prefer human-readable title, fall back to internal id
    #     title = str(doc.get("title") or doc.get("doc_id") or "Unknown source").strip()
    #     year = doc.get("year")
    #     court = (doc.get("court") or doc.get("category") or "").strip()

    #     meta_parts: List[str] = []
    #     if court:
    #         meta_parts.append(court)
    #     if year:
    #         meta_parts.append(str(year))

    #     if meta_parts:
    #         lines.append(f"{idx}. {title} ({', '.join(meta_parts)})")
    #     else:
    #         lines.append(f"{idx}. {title}")

    # return "\n".join(lines)


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
        from google import genai
        client = genai.Client(api_key=api_key)
        model_name = _get_gemini_model_name()
        resp = client.models.generate_content(model=model_name, contents=prompt)
        return getattr(resp, "text", str(resp))
    except Exception as e:
        return f"Error while generating draft: {e}"


# (Arguments panel / moot-court helpers now live in arguments_panel.py)

# ---------------------------------------------------------
# MAIN APP LOOP
# ---------------------------------------------------------

def main():
    st.set_page_config(page_title=APP_TITLE, page_icon="⚖️", layout="centered")
    _init_session()
    _inject_global_styles()

    # Lightweight, consistent header
    prefs = st.session_state.get("user_prefs", {})
    role_label = st.session_state.get("role") or "Not selected"
    jurisdiction_label = prefs.get("jurisdiction", "India")
    style_label = prefs.get("style") or "Default style"

    st.markdown(
        f"""
        <div class="lc-header">
            <div class="lc-header-title"></div>
            <p class="lc-header-subtitle">AI-assisted Indian legal research and drafting for students and professionals.</p>
            <div class="lc-status-row">
                <span class="lc-badge">Profile: {role_label.title() if isinstance(role_label, str) else role_label}</span>
                <span class="lc-badge">Jurisdiction: {jurisdiction_label}</span>
                <span class="lc-badge">Style: {style_label}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Sidebar controls for model & retrieval behaviour
    with st.sidebar:
        st.markdown("### Model Settings")

        # Map human-readable labels to actual embedding model IDs
        embedding_options = {
            "Gemini embedding (001)": "models/gemini-embedding-001",
            "Text embedding (004)": "models/text-embedding-004",
        }
        current_model = st.session_state.get("embedding_model", getattr(rag, "EMBEDDING_MODEL", "models/gemini-embedding-001"))
        # Find label corresponding to current model, default to first option
        default_label = next(
            (label for label, val in embedding_options.items() if val == current_model),
            list(embedding_options.keys())[0],
        )

        selected_label = st.selectbox(
            "Embedding model",
            list(embedding_options.keys()),
            index=list(embedding_options.keys()).index(default_label),
            help=(
                "Controls which Google Gemini embedding model is used when "
                "generating vectors for Pinecone RAG queries."
            ),
        )
        st.session_state["embedding_model"] = embedding_options[selected_label]

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
        # Ensure rag uses the embedding model selected in the UI
        try:
            rag.EMBEDDING_MODEL = st.session_state.get("embedding_model", rag.EMBEDDING_MODEL)
            log_cli("🔧 RAG", f"Using embedding model: {rag.EMBEDDING_MODEL}")
        except Exception:
            pass

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

        # 1) Case-law retrieval (Supreme Court / High Court)
        retrieved_docs = rag.search_chunks(
            query=user_input,
            top_k=5,
            court=court_filter,
            year_min=year_min,
            year_max=year_max,
            prefer_high_court=prefer_high_court,
        )

        # 2) Bare Acts retrieval – triggered when the query looks statute-heavy
        #    (e.g. "explain section 5 of the Limitation Act" or objective of a
        #    specific Act) or when the explicit intent is to simplify a section.
        need_bare_acts = False
        if analysis.get("intent") == "simplify_section":
            need_bare_acts = True
        else:
            statute_pattern = re.compile(
                r"(section\s*\d+[a-zA-Z]*|article\s*\d+[a-zA-Z]*|\bipc\b|\bi\.p\.c\b|\bcrpc\b|\bc\.r\.p\.c\b|evidence act|contract act|limitation act|bare act|[\w\s()/-]+ act,?\s*(19|20)\d{2})",
                re.IGNORECASE,
            )
            if statute_pattern.search(user_input):
                need_bare_acts = True

        # If no case-law context was found at all, fall back to Bare Acts
        # to still provide grounded statutory explanations where possible.
        if not retrieved_docs:
            need_bare_acts = True

        bare_act_docs: List[Dict[str, Any]] = []
        if need_bare_acts:
            bare_act_docs = rag.search_bare_acts(
                query=user_input,
                top_k=3,
            )
            if bare_act_docs:
                log_cli("📚 Bare Acts", f"Retrieved {len(bare_act_docs)} bare act chunks.")
                retrieved_docs = (retrieved_docs or []) + bare_act_docs

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

            # Aggregate retrieval confidence stats for CLI debugging
            scores = [float(d.get("score") or 0.0) for d in retrieved_docs]
            if scores:
                top_score = max(scores)
                min_score = min(scores)
                avg_score = sum(scores) / len(scores)
                log_cli(
                    "📈 RetrievalStats",
                    f"docs={len(retrieved_docs)} top={top_score:.3f} min={min_score:.3f} avg={avg_score:.3f}",
                )
        # --- Guardrail: corpus-limited disclaimer when clearly outside scope ---
        disclaimer = None
        # Detect whether we have any Bare Act context in the retrieved set.
        has_bare_acts = any(
            (d.get("court_level") == "Bare Act")
            or str(d.get("doc_type", "")).startswith("bare_act")
            for d in (retrieved_docs or [])
        )

        # For precise identity queries (explicit case_number + filing_year),
        # skip generic corpus-year warnings to avoid confusing filing vs
        # decision years. Also skip such warnings when Bare Acts are present,
        # because bare-acts corpus is not limited to 2005–2006.
        is_identity_query = bool(case_identity.get("case_number") and case_identity.get("filing_year"))
        intent_for_disclaimer = analysis.get("intent")
        if not is_identity_query and not has_bare_acts:
            # Known corpus window based on your ingestion of PDFs
            corpus_min, corpus_max = 2005, 2006
            query_year = year_min

            # Strong "no matching judgments" message should only be used
            # when the user is explicitly asking for judgments/citations and
            # we are clearly outside the corpus window.
            wants_specific_judgments = intent_for_disclaimer in {"find_judgements", "citation_finder", "case_summaries"}

            if not retrieved_docs and wants_specific_judgments and query_year and (
                query_year < corpus_min or query_year > corpus_max
            ):
                disclaimer = (
                    "⚠️ No matching judgments found in the current uploaded corpus for the year you mentioned. "
                    "The judgments corpus mainly covers 2005–2006. Please narrow the timeframe, "
                    "provide a different citation, or upload the relevant documents."
                )
            elif retrieved_docs and query_year and (
                query_year < corpus_min or query_year > corpus_max
            ):
                # Softer warning when we do have some matches, but year looks outside window.
                disclaimer = (
                    "⚠️ Your query mentions a year that may fall outside the primary "
                    "coverage of the current judgments corpus (2005–2006). Bare Acts "
                    "may still be available, but case-law results could be incomplete."
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
        # If we have a strong case identity (both number AND filing year), enforce
        # a hard match; otherwise fall back to normal semantic retrieval.
        # Using only a bare year or only a number is often too weak and can
        # incorrectly discard valid retrieved context.
        if case_identity.get("case_number") and case_identity.get("filing_year"):
            filtered_docs = [d for d in retrieved_docs if _doc_matches_identity(d, case_identity)]
            if not filtered_docs:
                answer = "The retrieved context does not match the specified case details."
                retrieved_docs = []  # Avoid leaking mismatched context downstream
            else:
                retrieved_docs = filtered_docs
                answer = _generate_answer(user_input, role, analysis["intent"], retrieved_docs)
        else:
            answer = _generate_answer(user_input, role, analysis["intent"], retrieved_docs)

        # Append structured citations block based on actual retrieved docs
        citation_block = _format_citation_block(retrieved_docs)
        if citation_block:
            answer = f"{answer}{citation_block}"

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

    # Global footer for the prototype
    st.markdown(
        """
        <div class="lc-footer">
            LexiCounsel is currently a prototype for educational and research support and does not constitute legal advice. 
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()