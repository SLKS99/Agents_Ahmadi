import streamlit as st
import json
import tools.socratic as socratic
from datetime import datetime

def initial_process(question: str):
    clarified_question = socratic.clarify_question(question)
    print(clarified_question)
    socratic_questions = socratic.socratic_pass(clarified_question)
    thoughts = socratic.tot_generation(socratic_questions, clarified_question)

    return clarified_question, socratic_questions, thoughts


@st.cache_resource
def init_session():
    if "interactions" not in st.session_state:
        st.session_state.interactions = []
    if "stage" not in st.session_state:
        st.session_state.stage = "initial"
    if "stop_hypothesis" not in st.session_state:
        st.session_state.stop_hypothesis = False
    return st.session_state

def insert_interaction(role, message, component):
    st.session_state.interactions.append({
        "role": role,
        "message": message,
        "component": component,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    })

def view_component(component):
    for i in st.session_state.interactions:
        if i["component"] == component:
            return i["message"]
    return None

def clear_conversation():
    st.session_state.interactions = []
    st.session_state.stage = "initial"
    st.toast("Conversation restarted")

def go_back_stage():
    if st.session_state.interactions:
        st.session_state.interactions.pop()
        st.toast("Returned to previous stage")
    else:
        st.warning("No previous stage to go back to.")

# --- FIX: Stop button now triggers a controlled rerun into hypothesis synthesis ---
def stop_and_create_hypothesis():
    st.session_state.stop_hypothesis = True
    st.session_state.stage = "hypothesis"
    st.toast("🧠 Generating hypothesis...")
    st.rerun()

def export_message_history():
    if not st.session_state.interactions:
        st.warning("No messages to export.")
        return None, None

    json_data = json.dumps(st.session_state.interactions, indent=2)
    text_data = "\n\n".join([
        f"[{i['timestamp']}] {i['role'].upper()} - {i['component'].upper()}: {i['message']}"
        for i in st.session_state.interactions
    ])
    return json_data, text_data


# ====== STREAMLIT UI ======
st.set_page_config(page_title="AI Hypothesis Agent", page_icon="✨", layout="centered")
st.title("AI Hypothesis Agent ✨")

init_session()

# --- Layout styling ---
st.markdown("""
<style>
    div[data-testid="stVerticalBlock"] div[data-testid="stHorizontalBlock"] {
        align-items: flex-end;
    }
    .bottom-container {
        position: fixed;
        bottom: 0;
        left: 0;
        width: 100%;
        background-color: white;
        border-top: 1px solid #ddd;
        padding: 0.8rem 1.5rem;
        box-shadow: 0 -2px 6px rgba(0,0,0,0.05);
        z-index: 999;
    }
</style>
""", unsafe_allow_html=True)

# --- Display existing chat ---
chat_container = st.container()
with chat_container:
    for i in st.session_state.interactions:
        with st.chat_message(i["role"]):
            st.markdown(i["message"])

st.markdown("<br><br><br><br>", unsafe_allow_html=True)

# --- Bottom Controls ---
bottom = st.container()
with bottom:
    st.markdown('<div class="bottom-container">', unsafe_allow_html=True)
    chat_col, options_col = st.columns([0.8, 0.2])

    with options_col:
        with st.popover("⚙️ Options"):
            st.markdown("### Conversation Controls")
            st.button("🔄 Restart", use_container_width=True, on_click=clear_conversation)
            st.button("🧠 Stop & Create Hypothesis", use_container_width=True, on_click=stop_and_create_hypothesis)
            st.button("⏪ Go Back", use_container_width=True, on_click=go_back_stage)
            st.markdown("---")
            st.markdown("### Export Data")

            json_data, text_data = export_message_history()
            if json_data and text_data:
                st.download_button(
                    label="💾 Export as JSON",
                    data=json_data,
                    file_name=f"conversation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json",
                    use_container_width=True
                )
                st.download_button(
                    label="📝 Export as Text",
                    data=text_data,
                    file_name=f"conversation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain",
                    use_container_width=True
                )

    st.markdown('</div>', unsafe_allow_html=True)

# ====== MAIN LOGIC FLOW ======
# --- FIX: If stop button was pressed, jump straight to hypothesis ---
if st.session_state.stop_hypothesis and st.session_state.stage != "analysis":
    with st.chat_message("assistant"):
        with st.spinner("Synthesizing hypothesis from current context..."):
            # Extract available data
            soc_q = view_component("retry_thinking_question") or view_component("clarified_question")
            picked = view_component("next_step_option_1") or view_component("first_thought_1")
            prev1 = view_component("next_step_option_2") or view_component("second_thought_1")
            prev2 = view_component("next_step_option_3") or view_component("third_thought_1")

            hypothesis = socratic.hypothesis_synthesis(soc_q, picked, prev1, prev2)

        st.markdown(hypothesis)
        st.success("🎉 Hypothesis generation complete (forced stop).")

    insert_interaction("assistant", hypothesis, "hypothesis")
    st.session_state.stop_hypothesis = False
    st.session_state.stage = "analysis"
    st.rerun()


# --- NORMAL STAGES ---
if st.session_state.stage == "initial":
    with chat_col:
        question = st.chat_input("Ask a question...")
    if question:
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                cl_question, soc_pass, thoughts_gen = initial_process(question)
                first_thought, second_thought, third_thought = thoughts_gen

            st.markdown(f"**Clarified Question:** {cl_question}")
            st.markdown(f"**Socratic Pass:** {soc_pass}")
            st.markdown("**Generated Thoughts:**")
            st.markdown(first_thought)
            st.markdown(second_thought)
            st.markdown(third_thought)

        insert_interaction("user", question, "initial_question")
        insert_interaction("assistant", cl_question, "clarified_question")
        insert_interaction("assistant", soc_pass, "socratic_pass")
        insert_interaction("assistant", first_thought, "first_thought_1")
        insert_interaction("assistant", second_thought, "second_thought_1")
        insert_interaction("assistant", third_thought, "third_thought_1")

        st.session_state.stage = "refine"
        st.rerun()

elif st.session_state.stage == "refine":
    with chat_col:
        user_choice = st.chat_input("Make a choice 1, 2, or 3...")
    if user_choice:
        if user_choice not in ["1", "2", "3"]:
            st.warning("Please enter 1, 2, or 3.")
            st.stop()

        with st.chat_message("user"):
            st.markdown(user_choice)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                if user_choice == "1":
                    picked = view_component("first_thought_1")
                    prev1 = view_component("second_thought_1")
                    prev2 = view_component("third_thought_1")
                elif user_choice == "2":
                    picked = view_component("second_thought_1")
                    prev1 = view_component("first_thought_1")
                    prev2 = view_component("third_thought_1")
                else:
                    picked = view_component("third_thought_1")
                    prev1 = view_component("first_thought_1")
                    prev2 = view_component("second_thought_1")

                initial_question = view_component("clarified_question")
                soc_q, options = socratic.retry_thinking_deepen_thoughts(
                    picked, prev1, prev2, initial_question
                )

            st.markdown(soc_q)
            for opt in options:
                st.markdown(opt)

        insert_interaction("user", user_choice, "tot_choice")
        insert_interaction("assistant", soc_q, "retry_thinking_question")
        insert_interaction("assistant", options[0], "next_step_option_1")
        insert_interaction("assistant", options[1], "next_step_option_2")
        insert_interaction("assistant", options[2], "next_step_option_3")

        st.session_state.stage = "hypothesis"
        st.rerun()

elif st.session_state.stage == "hypothesis":
    with chat_col:
        user_choice_2 = st.chat_input("Pick a next-step choice 1, 2, or 3...")
    if user_choice_2:
        if user_choice_2 not in ["1", "2", "3"]:
            st.warning("Please enter 1, 2, or 3.")
            st.stop()

        with st.chat_message("user"):
            st.markdown(user_choice_2)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                if user_choice_2 == "1":
                    picked = view_component("next_step_option_1")
                    prev1 = view_component("next_step_option_2")
                    prev2 = view_component("next_step_option_3")
                elif user_choice_2 == "2":
                    picked = view_component("next_step_option_2")
                    prev1 = view_component("next_step_option_1")
                    prev2 = view_component("next_step_option_3")
                else:
                    picked = view_component("next_step_option_3")
                    prev1 = view_component("next_step_option_1")
                    prev2 = view_component("next_step_option_2")

                soc_q = view_component("retry_thinking_question")
                hypothesis = socratic.hypothesis_synthesis(soc_q, picked, prev1, prev2)

            st.markdown(hypothesis)

        insert_interaction("user", user_choice_2, "retry_thinking_choice")
        insert_interaction("assistant", hypothesis, "hypothesis")

        st.success("🎉 Hypothesis generation complete!")
        st.session_state.stage = "analysis"
        st.rerun()

elif st.session_state.stage == "analysis":
    with st.chat_message("assistant"):
        with st.spinner("Analyzing Hypothesis and Producing Report..."):
            socratic_question = view_component("retry_thinking_question")
            hypothesis = view_component("hypothesis")

            analysis_rubric = socratic.local_hypothesis_analysis_fallback(hypothesis, socratic_question)

        st.markdown(analysis_rubric)

    insert_interaction("assistant", analysis_rubric, "analysis_rubric")
    st.success("🎉 Analysis complete!")
    st.session_state.stage = "initial"
    st.rerun()
