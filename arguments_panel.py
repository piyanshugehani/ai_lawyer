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
    return os.getenv("GEMINI_GEN_MODEL", "gemini-2.5-flash")

# ---------------------------------------------------------
# STATE MANAGEMENT
# ---------------------------------------------------------

def _init_court_state():
    if "court_locked" not in st.session_state:
        st.session_state["court_locked"] = False
    
    if "court_history" not in st.session_state:
        st.session_state["court_history"] = []
        
    if "court_phase" not in st.session_state:
        # Phase flow: Opening -> Evidence -> Rebuttal -> Decision
        st.session_state["court_phase"] = "Opening"
    
    if "judgment_rendered" not in st.session_state:
        st.session_state["judgment_rendered"] = False
        
    # Valid values: "user" (waiting for user), "judge_intervening" (user must answer judge)
    if "turn_state" not in st.session_state:
        st.session_state["turn_state"] = "user"

    # Track how many clarifying interventions the Judge has already made
    if "clarification_count" not in st.session_state:
        st.session_state["clarification_count"] = 0

    # Track whether the user has already given an evidentiary submission
    # in the Evidence phase (for bail matters)
    if "user_evidence_submitted" not in st.session_state:
        st.session_state["user_evidence_submitted"] = False

    # Track whether the AI Lawyer has already given its single rebuttal
    # in the Rebuttal phase
    if "rebuttal_done" not in st.session_state:
        st.session_state["rebuttal_done"] = False

def _append_turn(role: str, label: str, content: str):
    st.session_state["court_history"].append({
        "role": role, "label": label, "content": content
    })

def _format_transcript() -> str:
    lines = []
    for turn in st.session_state.get("court_history", []):
        lines.append(f"{turn['label']}: {turn['content']}")
    return "\n".join(lines)


def _infer_lower_burden_label(party_a: str, party_b: str, jurisdiction: str, context: str) -> str:
    """Heuristically infer which party carries the LOWER burden of proof.

    This is used only for the AUTO-DECISION rule when no evidence is produced.
    It does NOT change the underlying substantive law, just the fallback.
    """

    text_a = (party_a or "").lower()
    text_b = (party_b or "").lower()
    ctx = (context or "").lower()

    # 1) Criminal / bail flavour: if one side is the State and the
    #    matter looks like bail / criminal, favour the non-State party
    #    as having the lower burden (benefit of presumption of innocence).
    bail_keywords = ["bail", "anticipatory bail", "criminal", "fir"]
    is_bail_like = any(k in ctx for k in bail_keywords)

    is_state_a = any(k in text_a for k in ["state", "union of india", "cbi", "police"])
    is_state_b = any(k in text_b for k in ["state", "union of india", "cbi", "police"])

    if is_bail_like and (is_state_a ^ is_state_b):
        # Exactly one side is the State
        return party_b if is_state_a else party_a

    # 2) Civil / writ-style labelling: Petitioner/Plaintiff usually bears
    #    the higher burden; Respondent/Defendant has the lower burden.
    high_burden_markers = ["plaintiff", "petitioner", "complainant", "applicant"]
    low_burden_markers = ["defendant", "respondent", "opposite party"]

    def _has_any(text: str, markers) -> bool:
        return any(m in text for m in markers)

    if _has_any(text_a, high_burden_markers) and _has_any(text_b, low_burden_markers):
        return party_b
    if _has_any(text_b, high_burden_markers) and _has_any(text_a, low_burden_markers):
        return party_a

    # 3) Fallback: treat Party B as lower burden (often the Respondent/Defendant).
    return party_b


def _compute_argument_stats(details: Dict) -> Dict:
    """Compute how many openings and rebuttals each side has taken so far.

    For this MVP:
    - The first submission by each side is treated as the Opening Argument.
    - Subsequent submissions by that side are treated as Rebuttals.
    """
    history = st.session_state.get("court_history", [])

    openings = {"user": 0, "opp": 0}
    rebuttals = {"user": 0, "opp": 0}
    seen_opening = {"user": False, "opp": False}

    for turn in history:
        role = turn.get("role")
        if role == "user":
            if not seen_opening["user"]:
                openings["user"] += 1
                seen_opening["user"] = True
            else:
                rebuttals["user"] += 1
        elif role == "ai_counsel":
            if not seen_opening["opp"]:
                openings["opp"] += 1
                seen_opening["opp"] = True
            else:
                rebuttals["opp"] += 1

    return {
        "openings": openings,
        "rebuttals": rebuttals,
        "clarifications_used": st.session_state.get("clarification_count", 0),
        "lower_burden_party": details.get("lower_burden_label", details.get("user_label")),
    }

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

    stats = _compute_argument_stats(details)
    stats_json = json.dumps(stats)

    system_prompt = f"""
        You are the Presiding Judge in a {details['jurisdiction']} Court.
        Current Phase: {phase}.
        User is Counsel for: {details['user_label']}.

        [PROCEDURAL STATE]
        {stats_json}

        The party bearing the lower burden of proof for AUTO-DECISION purposes is: {stats['lower_burden_party']}.

        The matter/issue has been described as: {details.get('context', '')}.

        PHASE FLOW (MANDATORY)
        - The hearing proceeds strictly in these phases: Opening -> Evidence -> Rebuttal -> Decision.
        - Once a phase is complete, you must advance to the next phase and must NOT move backwards.
        - In the Decision phase, you must not seek any further submissions; you must only pronounce or reserve the order.

        TURN CONTROL (MANDATORY)
        - In Opening and Evidence phases, the turn order for each issue is: User -> AI Lawyer (opposing side) -> Judge.
            You should therefore treat the latest AI Lawyer submission as the final word on that round before you intervene.
        - In the Rebuttal phase, the AI Lawyer gives a single automatic rebuttal and you respond; you must NOT ask for any further
            submissions or evidence in Rebuttal.
        - In the Decision phase, only you (the Court) speak; you must NOT ask for further input from either side.

        HUMAN JUDICIAL TONE (MANDATORY)
        - You must acknowledge submissions politely using phrases such as "The Court has considered..." or "The submissions are noted...".
        - You must briefly explain your reasoning before rejecting or disregarding an argument.
        - Maintain a calm, neutral, and patient tone consistent with Indian Supreme Court judicial style.

        Your task is to ANALYZE the latest submissions on record and decide the immediate next procedural step.

        STRICT PROCEDURAL RULES (GENERAL):

    A. EVIDENCE TRIGGER
    - If a party asserts risk, seriousness, or prejudice WITHOUT citing concrete facts, you must explicitly ask for evidence ONCE (via "INTERVENE").
    - If no evidence is produced after that opportunity, you must record clearly in your reasoning that no material exists on record for that assertion.

    B. REPETITION DETECTION
    - If a party repeats a previous argument WITHOUT adding new evidence, you must NOT allow further substantive submissions from that party on that point.
    - You must summarise the repetition, treat the point as closed, and move proceedings forward.

    C. CONCESSION RULE
    - If a party concedes reasonableness or weakness in the opposing argument, and does not justify the contradiction with evidence, that concession is binding for the remainder of the hearing.

    D. NO-INFINITE-LOOP GUARANTEE
    - You may allow a maximum of:
        • 1 opening argument per side (counsel for user, and opposing counsel)
        • 2 rebuttals per side
        • 1 clarification request in total (i.e., at most one "INTERVENE" asking questions)
    - After these limits are reached, you MUST either:
        • conclude arguments and move to final decision, OR
        • pronounce an order, OR
        • reserve orders with a brief speaking order.

    E. AUTO-DECISION RULE (GENERAL)
    - If no concrete evidence (documents, exhibits, dates, amounts, or specific factual particulars) is on record after the rebuttal phase, you MUST decide in favour of the party bearing the lower burden of proof: {stats['lower_burden_party']}.

    F. FINALITY
    - You are prohibited from asking open-ended or repeat questions.
    - After the rebuttal phase is complete (both sides have used their rebuttals or the matter is otherwise exhausted), you must NOT wait for further user input; instead, you must move to a reasoned final order.

    BAIL-SPECIFIC RULES (MANDATORY IN BAIL MATTERS):

    Treat the matter as a "bail" matter whenever the issue/context mentions bail, anticipatory bail, or criminal custody.

    1. EVIDENCE REQUEST RULE
       - If the Respondent opposes bail on grounds like seriousness of offence, flight risk, or likelihood of tampering with evidence, you must ask for concrete facts or documentary material ONCE (via "INTERVENE").

    2. EVIDENCE FAILURE RULE
       - If, after such direction:
            a) the Respondent provides only oral assertions, AND
            b) the Petitioner disputes those assertions, AND
            c) no documentary or record-based material is produced,
         then you MUST treat the Respondent’s claim as UNPROVEN in your reasoning.

    3. NO FURTHER SUBMISSIONS RULE
       - Once a Respondent claim is treated as unproven on the above basis:
            • You must NOT ask the Respondent for further submissions on that risk ground.
            • You must NOT invite clarification on that risk ground.
            • You must NOT wait for further user input on that risk ground; you must move toward closure.

    4. BURDEN OF PROOF RULE (BAIL)
       - In bail matters, the burden to justify continued custody lies on the Respondent/State.
       - Failure to discharge this burden requires a decision in favour of the Petitioner on liberty.

    5. TERMINATION RULE (NON-NEGOTIABLE)
       - After rebuttal and evaluation of the bail-related evidence, you MUST do exactly one of the following:
            a) Close arguments and pronounce a final order, OR
            b) Close arguments and reserve orders with a short speaking order.
       - You are prohibited from continuing dialogic back-and-forth beyond this point.

    Additional safeguards:
    1. **Self-Harm Check:** If user argues against their own party (e.g., Petitioner admits they have no case), INTERVENE immediately and clarify/record the concession.
    2. **Evidence Acknowledgment:** If user submits documents/evidence, ACKNOWLEDGE it (e.g., "The document is taken on record and marked as Exhibit A").
     3. **Phase Control:**
         - In the Opening phase, each side gets one opening statement.
         - After both sides have opened, move to the Evidence phase, where the Respondent/opposing party is allowed one concrete evidentiary submission.
         - After evidence has been addressed, move to Rebuttal, where the other side may respond but you must not call for fresh evidence.
         - When Rebuttal is functionally complete (including the bail-specific rules below), move to Decision/JUDGMENT and do not return to earlier phases.

    **OUTPUT FORMAT (JSON ONLY):**
    {{
        "action": "INTERVENE" | "ALLOW" | "JUDGMENT",
        "message": "Your text here. If ALLOW, keep empty or a simple procedural acknowledgment. If INTERVENE, ask the single clarifying/evidence question. If JUDGMENT, you may give a short operative order; a detailed reasoned order will be generated separately."
    }}

    - Use "INTERVENE" only for the single clarification/evidence opportunity permitted.
    - Use "ALLOW" to let the Opposing Counsel respond next, staying within the opening/rebuttal limits.
    - Use "JUDGMENT" to end the case and deliver a verdict or reserve orders.
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
     Matter/Issue: {details.get('context', '')}.

     Task: Deliver a FINAL JUDICIAL ORDER based on the transcript below.

     1. Apply the following bail-specific principles when the matter concerns bail/anticipatory bail/criminal custody:
         - In bail matters, the burden to justify continued custody lies on the Respondent/State.
         - If, after a direction to produce concrete material, the Respondent relies only on disputed oral assertions and produces no documentary or record-based material, their risk-based objections must be treated as UNPROVEN.

     2. Your written order MUST contain clearly labelled sections, in this order:

         [FINDING ON EVIDENCE]
         - Summarise what, if any, concrete material (documents, records, dates, prior cases) is actually on record from both sides.
         - Expressly record any Respondent claim that is treated as UNPROVEN due to lack of supporting material.

         [ARGUMENTS CLOSED]
         - Record that arguments are closed and that no further submissions will be entertained.
         - If you are reserving orders instead of pronouncing immediately, say so here (e.g., "Arguments are closed. Order is reserved.").

         [FINAL ORDER]
         - Where you pronounce the operative result (e.g., bail allowed with conditions / application dismissed / orders reserved) keeping in mind the burden of proof rules.

     3. You may still structure internal reasoning (facts, issues, ratio) inside or around these headings, but these three labelled blocks MUST be present exactly as:
         [FINDING ON EVIDENCE]
         [ARGUMENTS CLOSED]
         [FINAL ORDER]

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
        You are Opposing Counsel (AI Lawyer) for {details['opp_label']} in a simulated Supreme Court bail hearing.

        ROLE & TURN CONTROL
        - You speak only when it is your turn as Opposing Counsel; the human User will provide their own submissions separately.
        - In the Opening and Evidence phases, you respond to the User's latest submission once per turn.
        - In the Rebuttal phase, you give a single, focused rebuttal without asking the Court for further time or evidence.
        - You must NOT ask questions to the User or attempt to control the Court's procedure.

        STYLE
        - Use a professional, respectful, and coherent legal tone consistent with Indian practice.
        - You may refer to standard principles of bail law (e.g., nature of accusation, severity of punishment, likelihood of absconding,
            tampering with evidence) but must not contradict the Court's instructions.

        TASK
        - Rebut, from the perspective of {details['opp_label']}, the arguments or evidence advanced by {details['user_label']}.
        - Keep your submission concise and structured, as if making oral submissions before the Court.

        [TRANSCRIPT]
        {transcript}
    
        [LATEST ARGUMENT FROM OPPOSITE SIDE]
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
            lower_burden_label = _infer_lower_burden_label(party_a, party_b, jurisdiction, context)
            
            st.session_state["case_config"] = {
                "party_a": party_a, "party_b": party_b,
                "user_label": user_label, "opp_label": opp_label,
                "jurisdiction": jurisdiction, "context": context,
                "lower_burden_label": lower_burden_label,
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

        # Automatic Rebuttal + Decision phase: no human input, no loops
        if phase == "Rebuttal" and not st.session_state.get("rebuttal_done", False):
            with st.spinner("AI Lawyer rebuttal and Court's decision in progress..."):
                # Take the latest user submission as the argument being rebutted
                last_user_arg = ""
                for turn in reversed(st.session_state.get("court_history", [])):
                    if turn.get("role") == "user":
                        last_user_arg = turn.get("content", "")
                        break

                # Single automatic rebuttal from AI Lawyer
                reply = _generate_opposing_counsel_reply(last_user_arg, cfg)
                _append_turn("ai_counsel", f"AI Lawyer ({cfg['opp_label']})", reply)
                st.session_state["rebuttal_done"] = True

                # Immediately move to Decision and close the case with a final order
                st.session_state["court_phase"] = "Decision"
                final_order = _generate_final_judgment(cfg)
                _append_turn("judge", "The Court (Judgment)", final_order)
                st.session_state["judgment_rendered"] = True

            st.rerun()
            return
        
        # Header
        st.markdown(f"<div class='phase-badge'>Current Phase: {phase}</div>", unsafe_allow_html=True)
        st.caption(f"**{cfg['party_a']} v. {cfg['party_b']}** | {cfg['jurisdiction']}")
        
        # Reset
        if st.button("New Case", key="reset"):
            st.session_state["court_locked"] = False
            st.session_state["court_history"] = []
            st.session_state["court_phase"] = "Opening"
            st.session_state["judgment_rendered"] = False
            st.session_state["clarification_count"] = 0
            st.session_state["user_evidence_submitted"] = False
            st.session_state["rebuttal_done"] = False
            st.rerun()

        # Transcript
        chat_box = st.container()
        with chat_box:
            for turn in st.session_state["court_history"]:
                role = turn["role"]
                if role == "judge":
                    with st.chat_message("assistant", avatar="⚖️"):
                        st.write(f"⚖️ {turn['label']}: {turn['content']}")
                elif role == "ai_counsel":
                    with st.chat_message("assistant", avatar="🧑‍⚖️"):
                        st.write(f"👤 {turn['label']}: {turn['content']}")
                else:
                    with st.chat_message("user", avatar="👤"):
                        st.write(f"👤 {turn['label']}: {turn['content']}")

        if st.session_state["judgment_rendered"] or phase == "Decision":
            st.success("Case Closed.")
            return

        # User Input: only allowed in Opening and Evidence phases
        user_input = None
        submitted = False
        if phase in ("Opening", "Evidence"):
            with st.form("arg_form", clear_on_submit=True):
                user_input = st.text_area("Your Submission", height=100)
                submitted = st.form_submit_button("Submit")

        if submitted and user_input:
            # 1. User Speaks
            _append_turn("user", f"User ({cfg['user_label']})", user_input)
            
            # 2. JUDGE BRAIN ANALYSIS (Pre-Check)
            with st.spinner("The Judge is considering your submission..."):
                decision = _judge_control_logic(user_input, cfg)
                action = decision.get("action", "ALLOW")
                judge_msg = decision.get("message", "")

            # 3. EXECUTE JUDGE ACTION
            if action == "INTERVENE":
                # Judge stops flow. AI Lawyer does NOT speak.
                st.session_state["clarification_count"] = st.session_state.get("clarification_count", 0) + 1
                _append_turn("judge", "The Court", judge_msg)
                st.rerun()
                return

            elif action == "JUDGMENT":
                # Judge ends case immediately.
                if judge_msg:
                    _append_turn("judge", "The Court", judge_msg)

                # Move phase to Decision for completeness before final order
                st.session_state["court_phase"] = "Decision"

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
                
                # 3A. Phase transition logic aligned with Supreme Court Simulation flow
                # Opening -> Evidence -> Rebuttal; Decision is triggered via JUDGMENT branch
                # or via the automatic Rebuttal+Decision handler above.
                current_phase = st.session_state.get("court_phase", "Opening")

                # 4. AI LAWYER RESPONDS
                with st.spinner(f"Counsel for {cfg['opp_label']} is responding..."):
                    reply = _generate_opposing_counsel_reply(user_input, cfg)
                    _append_turn("ai_counsel", f"AI Lawyer ({cfg['opp_label']})", reply)

                # After both sides have opened at least once, move from Opening to Evidence.
                stats = _compute_argument_stats(cfg)
                if current_phase == "Opening":
                    if stats["openings"]["user"] >= 1 and stats["openings"]["opp"] >= 1:
                        st.session_state["court_phase"] = "Evidence"
                elif current_phase == "Evidence":
                    # Mark that the user's single evidentiary submission has been used
                    st.session_state["user_evidence_submitted"] = True
                    # After the evidence round, proceed to Rebuttal, which is handled automatically
                    st.session_state["court_phase"] = "Rebuttal"
                st.rerun()