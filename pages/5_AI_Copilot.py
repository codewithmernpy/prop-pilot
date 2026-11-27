"""Page 5: AI Copilot & Offline Deterministic Plan Explainer."""

from __future__ import annotations

import streamlit as st

from proppilot.ai import PlanSession, generate_offline_plan_explanation, run_copilot_query
from proppilot.ui import cached_optimise, init_page_and_sidebar, render_suite_footer

ctx = init_page_and_sidebar("AI Copilot")
dataset = ctx["dataset"]
rules = ctx["rules"]

# Ensure we have a fast cached optimisation result ready for both offline explainer and AI session.
opt_res = cached_optimise(
    ctx["source_key"],
    ctx["raw_bytes"],
    ctx["file_ext"],
    ctx["rules_yaml"],
    objective="fastest",
    max_fail=0.20,
    full_precision=False,
)

if "pp_copilot_tool_log" not in st.session_state:
    st.session_state["pp_copilot_tool_log"] = []
if "pp_copilot_messages" not in st.session_state:
    st.session_state["pp_copilot_messages"] = []

session = PlanSession(
    dataset=dataset,
    rules=rules,
    objective=opt_res.objective,
    max_fail=opt_res.max_fail,
    n_bootstrap=400,
    seed=42,
    max_lose_funded=opt_res.max_lose_funded,
    opt_result=opt_res,
)

st.title("5. AI Risk Copilot")
if ctx["api_key"]:
    st.caption("Anthropic API key active in browser session — Claude tool_runner enabled.")
else:
    st.info(
        "No API key entered — running in deterministic offline mode. "
        "Enter an Anthropic key in the sidebar (session only) to chat with Claude tool_runner."
    )

with st.expander("Deterministic Offline Plan Brief (Always Available)", expanded=True):
    st.markdown(generate_offline_plan_explanation(session))

st.subheader("Copilot Chat")
q1, q2, q3 = st.columns(3)
quick_prompt: str | None = None
if q1.button("Why do challenges fail?", key="cp_q_fail"):
    quick_prompt = "Why do challenges fail on the validation data, and which correlated days drive breaches?"
if q2.button("Explain funded payouts & CPPI", key="cp_q_funded"):
    quick_prompt = "Explain the 12-month funded projection, keep buffer, and CPPI sizing risk."
if q3.button("Compare fastest vs cheapest", key="cp_q_cmp"):
    quick_prompt = "Compare fastest vs cheapest objectives on this strategy and highlight any overfit risk."

for msg in st.session_state["pp_copilot_messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

typed_prompt = st.chat_input("Ask PropPilot Copilot about sizing, failures, or rules...")
active_prompt = quick_prompt or typed_prompt

if active_prompt:
    st.session_state["pp_copilot_messages"].append({"role": "user", "content": active_prompt})
    with st.chat_message("user"):
        st.markdown(active_prompt)

    reply = run_copilot_query(
        session=session,
        prompt=active_prompt,
        history=st.session_state["pp_copilot_messages"][:-1],
        api_key=ctx["api_key"],
    )
    if session.tool_log:
        st.session_state["pp_copilot_tool_log"].extend(session.tool_log)

    assistant_text = reply.text
    if reply.fallback_reason and ctx["api_key"]:
        assistant_text = f"*{reply.fallback_reason}*\n\n{assistant_text}"

    st.session_state["pp_copilot_messages"].append({"role": "assistant", "content": assistant_text})
    with st.chat_message("assistant"):
        st.markdown(assistant_text)

with st.expander("Tool-Call Audit Log", expanded=False):
    logs = st.session_state["pp_copilot_tool_log"]
    if logs:
        st.dataframe(logs, width="stretch", hide_index=True)
    else:
        st.caption("No AI tool calls executed yet in this session.")

render_suite_footer()
