from __future__ import annotations
import os
import json
import streamlit as st
from typing import Dict, List, Optional

# ---------------------------------------------------------
# CONFIG & HELPERS
# ---------------------------------------------------------

def _get_gemini_api_key() -> Optional[str]:
    key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if key: return key
    try: return st.secrets.get("GEMINI_API_KEY")
    except: return None

def _get_model_name() -> str:
    return os.getenv("GEMINI_GEN_MODEL", "gemini-1.5-flash")

# ---------------------------------------------------------
# STATE MANAGEMENT
# ---------------------------------------------------------

def _init_court_state():
    if "court_locked" not in st.session_state:
        st.session_state["court_locked"] = False
    
    if "court_history" not in st.session_state:
        st.session_state["court_history"] = []
        
    if "court_phase" not in st.session_state:
        st.session_state["court_phase"] = "Opening Statement" # Opening -> Rebuttal -> Judgment
    
    if "judgment_rendered" not in st.session_state:
        st.session_state["judgment_rendered"] = False
        
    # Valid values: "user" (waiting for user), "judge_intervening" (user must answer judge)
    if "turn_state" not in st.session_state:
        st.session_state["turn_state"] = "user"

def _append_turn(role: str, label: str, content: str):
    st.session_state["court_history"].append({
        "role": role, "label": label, "content": content
    })

def _format_transcript() -> str:
    lines = []
    for turn in st.session_state.get("court_history", []):
        lines.append(f"{turn['label']} ({turn['role']}): {turn['content']}")
    return "\n".join(lines)

# ---------------------------------------------------------
# THE JUDGE BRAIN (CONTROLLER)
# ---------------------------------------------------------

def _judge_control_logic(user_input: str, details: Dict) -> Dict:
    """
    Decides the flow of the court. 
    Returns JSON: { "action": "INTERVENE"|"ALLOW"|"JUDGMENT", "message": "..." }
    """
    api_key = _get_gemini_api_key()
    if not api_key: return {"action": "ALLOW", "message": ""}

    transcript = _format_transcript()
    phase = st.session_state["court_phase"]
    
    system_prompt = f"""
    You are the Presiding Judge in a {details['jurisdiction']} Court.
    Current Phase: {phase}.
    User is Counsel for: {details['user_label']}.
    
    Your task is to ANALYZE the user's latest argument and decide the immediate next step.
    
    **CRITICAL RULES:**
    1. **Self-Harm Check:** If user argues against their own party (e.g., Petitioner admits they have no case), INTERVENE immediately.
    2. **Evidence:** If user submits documents/evidence, ACKNOWLEDGE it ("Marked as Exhibit A").
    3. **Phase Control:** - If arguments seem repetitive or complete, move to JUDGMENT.
       - If phase is 'Opening' and user is done, allow transition to 'Rebuttal'.
    4. **Intervention:** If user is unclear or contradictory, ask a CLARIFYING question.
    
    **OUTPUT FORMAT (JSON ONLY):**
    {{
        "action": "INTERVENE" | "ALLOW" | "JUDGMENT",
        "message": "Your text here. If ALLOW, keep empty or simple acknowledgment. If INTERVENE, ask the question."
    }}
    
    - Use "INTERVENE" to STOP the Opposing Counsel from speaking (User must answer you first).
    - Use "ALLOW" to let the Opposing Counsel respond next.
    - Use "JUDGMENT" to end the case and deliver verdict.
    """
    
    prompt = f"""
    {system_prompt}
    
    [TRANSCRIPT]
    {transcript}
    
    [USER'S LATEST SUBMISSION]
    {user_input}
    
    [JSON DECISION]
    """
    
    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(_get_model_name(), generation_config={"response_mime_type": "application/json"})
        response = model.generate_content(prompt)
        return json.loads(response.text)
    except Exception as e:
        print(f"Judge Error: {e}")
        return {"action": "ALLOW", "message": ""}

def _generate_final_judgment(details: Dict) -> str:
    """Generates the final reasoned order."""
    api_key = _get_gemini_api_key()
    transcript = _format_transcript()
    
    prompt = f"""
    Role: Judge ({details['jurisdiction']}).
    Task: Deliver FINAL JUDGMENT based on the transcript below.
    Structure:
    1. Introduction of Parties.
    2. Summary of Arguments.
    3. Reasoning (Ratio Decidendi).
    4. Final Order (Allowed/Dismissed).
    
    [TRANSCRIPT]
    {transcript}
    """
    
    import google.generativeai as genai
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(_get_model_name())
    return model.generate_content(prompt).text

# ---------------------------------------------------------
# OPPOSING COUNSEL (AI LAWYER)
# ---------------------------------------------------------

def _generate_opposing_counsel_reply(user_input: str, details: Dict) -> str:
    api_key = _get_gemini_api_key()
    transcript = _format_transcript()
    
    prompt = f"""
    You are Opposing Counsel for {details['opp_label']}.
    Rebut the arguments made by {details['user_label']} based on the transcript.
    Be sharp, professional, and concise.
    
    [TRANSCRIPT]
    {transcript}
    
    [LATEST ARGUMENT]
    {user_input}
    """
    
    import google.generativeai as genai
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(_get_model_name())
    return model.generate_content(prompt).text

# ---------------------------------------------------------
# UI RENDERER
# ---------------------------------------------------------

def render_arguments_panel():
    _init_court_state()

    # CSS for Phase Indicator
    st.markdown("""
    <style>
    .phase-badge {
        background-color: #d4edda; color: #155724; padding: 5px 10px; 
        border-radius: 15px; font-weight: bold; font-size: 0.8rem;
        border: 1px solid #c3e6cb; margin-bottom: 10px; display: inline-block;
    }
    .judge-alert {
        background-color: #fff3cd; color: #856404; padding: 10px;
        border-left: 5px solid #ffeeba; margin: 10px 0;
    }
    </style>
    """, unsafe_allow_html=True)

    # -----------------------------------------------------
    # SETUP PHASE
    # -----------------------------------------------------
    if not st.session_state["court_locked"]:
        st.subheader("⚖️ Configure Court Session")
        c1, c2 = st.columns(2)
        with c1:
            party_a = st.text_input("Party A", "Mr. X (Petitioner)")
            party_b = st.text_input("Party B", "State (Respondent)")
        with c2:
            jurisdiction = st.selectbox("Jurisdiction", ["Supreme Court", "High Court", "Trial Court"])
            context = st.text_input("Issue", "Bail Application")
            
        user_side = st.radio("Representing:", ["Party A", "Party B"], horizontal=True)
        
        if st.button("🏛️ Call Case", type="primary"):
            user_label = party_a if user_side == "Party A" else party_b
            opp_label = party_b if user_side == "Party A" else party_a
            
            st.session_state["case_config"] = {
                "party_a": party_a, "party_b": party_b,
                "user_label": user_label, "opp_label": opp_label,
                "jurisdiction": jurisdiction, "context": context
            }
            
            # Judge Opens
            opening = f"The Court is in session. Matter of {party_a} v. {party_b}. Counsel for {user_label}, proceed with Opening Arguments."
            _append_turn("judge", "The Court", opening)
            
            st.session_state["court_locked"] = True
            st.rerun()

    # -----------------------------------------------------
    # HEARING PHASE
    # -----------------------------------------------------
    else:
        cfg = st.session_state["case_config"]
        phase = st.session_state["court_phase"]
        
        # Header
        st.markdown(f"<div class='phase-badge'>Current Phase: {phase}</div>", unsafe_allow_html=True)
        st.caption(f"**{cfg['party_a']} v. {cfg['party_b']}** | {cfg['jurisdiction']}")
        
        # Reset
        if st.button("New Case", key="reset"):
            st.session_state["court_locked"] = False
            st.session_state["court_history"] = []
            st.session_state["court_phase"] = "Opening Statement"
            st.session_state["judgment_rendered"] = False
            st.rerun()

        # Transcript
        chat_box = st.container()
        with chat_box:
            for turn in st.session_state["court_history"]:
                role = turn["role"]
                if role == "judge":
                    with st.chat_message("assistant", avatar="⚖️"):
                        st.write(f"**{turn['label']}:** {turn['content']}")
                elif role == "ai_counsel":
                    with st.chat_message("assistant", avatar="🧑‍⚖️"):
                        st.write(f"**{turn['label']}:** {turn['content']}")
                else:
                    with st.chat_message("user", avatar="👤"):
                        st.write(f"**{turn['label']}:** {turn['content']}")

        if st.session_state["judgment_rendered"]:
            st.success("Case Closed.")
            return

        # User Input
        with st.form("arg_form", clear_on_submit=True):
            user_input = st.text_area("Your Submission", height=100)
            submitted = st.form_submit_button("Submit")

        if submitted and user_input:
            # 1. User Speaks
            _append_turn("user", f"Counsel for {cfg['user_label']}", user_input)
            
            # 2. JUDGE BRAIN ANALYSIS (Pre-Check)
            with st.spinner("The Judge is considering your submission..."):
                decision = _judge_control_logic(user_input, cfg)
                action = decision.get("action", "ALLOW")
                judge_msg = decision.get("message", "")

            # 3. EXECUTE JUDGE ACTION
            if action == "INTERVENE":
                # Judge stops flow. AI Lawyer does NOT speak.
                _append_turn("judge", "The Court", judge_msg)
                st.rerun()
                return

            elif action == "JUDGMENT":
                # Judge ends case immediately.
                if judge_msg:
                    _append_turn("judge", "The Court", judge_msg)
                
                with st.spinner("Writing Judgment..."):
                    final_order = _generate_final_judgment(cfg)
                    _append_turn("judge", "The Court (Judgment)", final_order)
                    st.session_state["judgment_rendered"] = True
                st.rerun()
                return

            else: # ACTION == "ALLOW"
                # Judge might acknowledge evidence but lets flow continue
                if judge_msg:
                    _append_turn("judge", "The Court", judge_msg)
                
                # Check phase transition logic (Simple toggle for MVP)
                if st.session_state["court_phase"] == "Opening Statement":
                    st.session_state["court_phase"] = "Rebuttal"
                
                # 4. AI LAWYER RESPONDS
                with st.spinner(f"Counsel for {cfg['opp_label']} is responding..."):
                    reply = _generate_opposing_counsel_reply(user_input, cfg)
                    _append_turn("ai_counsel", f"Counsel for {cfg['opp_label']}", reply)
                st.rerun()