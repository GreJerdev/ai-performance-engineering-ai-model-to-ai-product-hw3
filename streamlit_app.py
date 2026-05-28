import os
import sqlite3
import uuid

import streamlit as st
from dotenv import load_dotenv
from langgraph.checkpoint.sqlite import SqliteSaver

from app.ai_agent.agent import LangGraphAgent, get_llm

load_dotenv()

CHECKPOINT_DB = "checkpoints.sqlite"

AGENT = None

@st.cache_resource
def get_agent() -> LangGraphAgent:
    global AGENT
    if not AGENT:
        """Create one shared agent instance for the Streamlit process."""
        conn = sqlite3.connect(CHECKPOINT_DB, check_same_thread=False)
        checkpointer = SqliteSaver(conn)
        AGENT = LangGraphAgent(get_llm(), checkpointer=checkpointer)

    return AGENT


def _response_to_text(response: object) -> str:
    if response is None:
        return ""
    if isinstance(response, str):
        return response
    return str(response)


def main() -> None:
    
    st.set_page_config(page_title="Bitext Agent", page_icon="🤖", layout="centered")
    st.title("Bitext AI Agent")
    st.caption("Ask questions about cancellation, delivery, orders, and payment intents.")

    if "thread_id" not in st.session_state:
        st.session_state.thread_id = f"streamlit-{uuid.uuid4()}"
    if "messages" not in st.session_state:
        st.session_state.messages = []

    with st.sidebar:
        st.subheader("Session")
        st.text_input("Thread ID", key="thread_id")
        if st.button("Clear chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    prompt = st.chat_input("Type your message")
    if not prompt:
        return

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                agent = get_agent()
                
                config = {"configurable": {"thread_id": st.session_state.thread_id}}

                answer = _response_to_text(agent.invoke(prompt, config=config))
            except Exception as exc:  # noqa: BLE001
                answer = f"Error: {exc}"
        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()
