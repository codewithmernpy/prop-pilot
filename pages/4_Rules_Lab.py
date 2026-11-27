"""Page 4: Rules Lab — AI Rule Extractor & Offline Manual YAML Editor."""

from __future__ import annotations

import streamlit as st

from proppilot.ai import RuleExtractionError, extract_rules_from_text, extract_rules_offline
from proppilot.rules import PhaseRule, PropRules, parse_rules_yaml_text
from proppilot.ui import init_page_and_sidebar, render_suite_footer

ctx = init_page_and_sidebar("Rules Lab")
rules = ctx["rules"]

st.title("4. Rules Lab (AI Extractor & Offline YAML Editor)")
st.caption(
    "Paste unstructured rules text to extract PropRules with Claude (flagging ambiguities), "
    "or edit rule parameters manually offline."
)

tab_ai, tab_manual = st.tabs(["AI Rule Extraction", "Offline Manual Rule Editor"])

with tab_ai:
    sample_text = (
        "2-Phase Evaluation: Phase 1 profit target 8%, Phase 2 profit target 5%. "
        "Minimum 4 trading days per phase. Daily loss limit 5% of day-start balance resetting at 00:00 UTC. "
        "Maximum overall static drawdown 10% of initial balance. Fee $500 per $100,000 account, "
        "80% profit split, payouts every 14 days."
    )
    raw_text = st.text_area(
        "Paste prop-firm rules page text",
        value=st.session_state.get("rl_raw_text", sample_text),
        height=160,
        key="rl_raw_text",
    )
    if st.button("Extract Rules", key="rl_extract_btn"):
        extracted = None
        try:
            extracted = extract_rules_from_text(raw_text, api_key=ctx["api_key"])
            st.success("Extracted PropRules YAML with Claude. Review ambiguities below before applying.")
        except RuleExtractionError as exc:
            try:
                extracted = extract_rules_offline(raw_text)
                st.warning(
                    f"AI extraction unavailable ({exc}). Used the offline keyword parser instead; "
                    "review every value and the ambiguity list before applying."
                )
            except ValueError as off_exc:
                st.error(f"Offline parser could not build valid rules: {off_exc}")
        except ValueError as exc:
            st.error(f"Extraction error: {exc}")
        if extracted is not None:
            st.session_state["rl_extracted_yaml"] = extracted.to_yaml()
            st.session_state["rl_ambiguities"] = list(extracted.ambiguities)

    if "rl_ambiguities" in st.session_state:
        ambs = st.session_state["rl_ambiguities"]
        if ambs:
            st.warning("Flagged Ambiguities (confirm before running):")
            for item in ambs:
                st.markdown(f"- {item}")
        else:
            st.success("No ambiguities flagged by extractor.")

    if "rl_extracted_yaml" in st.session_state:
        st.code(st.session_state["rl_extracted_yaml"], language="yaml")
        if st.button("Apply Extracted Rules to Session", key="rl_apply_ai_btn"):
            st.session_state["pp_custom_rules_yaml"] = st.session_state["rl_extracted_yaml"]
            st.session_state["pp_pending_rules_mode"] = "Custom (Rules Lab)"
            st.success("Applied extracted rules to session. Select 'Custom (Rules Lab)' in the sidebar.")

with tab_manual:
    st.subheader("Structured Rule Form (100% Offline)")
    c1, c2, c3 = st.columns(3)
    with c1:
        r_name = st.text_input("Rulebook Name", value=rules.name, key="rl_name")
        n_phases = st.selectbox(
            "Number of evaluation phases",
            options=[0, 1, 2],
            index=min(2, len(rules.phases)),
            key="rl_n_phases",
        )
        p1_target = st.number_input(
            "Phase 1 Target (%)",
            min_value=1.0,
            max_value=50.0,
            value=float(rules.phases[0].target_pct) if len(rules.phases) >= 1 else 8.0,
            step=0.5,
            key="rl_p1_tgt",
            disabled=(n_phases == 0),
        )
        p1_min_days = st.number_input(
            "Phase 1 Min Trading Days",
            min_value=0,
            max_value=60,
            value=int(rules.phases[0].min_days) if len(rules.phases) >= 1 else 4,
            step=1,
            key="rl_p1_min",
            disabled=(n_phases == 0),
        )
    with c2:
        p2_target = st.number_input(
            "Phase 2 Target (%)",
            min_value=1.0,
            max_value=50.0,
            value=float(rules.phases[1].target_pct) if len(rules.phases) >= 2 else 5.0,
            step=0.5,
            key="rl_p2_tgt",
            disabled=(n_phases < 2),
        )
        p2_min_days = st.number_input(
            "Phase 2 Min Trading Days",
            min_value=0,
            max_value=60,
            value=int(rules.phases[1].min_days) if len(rules.phases) >= 2 else 4,
            step=1,
            key="rl_p2_min",
            disabled=(n_phases < 2),
        )
        daily_loss = st.number_input(
            "Daily Loss Limit (%)",
            min_value=0.5,
            max_value=25.0,
            value=float(rules.daily_loss_pct),
            step=0.5,
            key="rl_daily_loss",
        )
        daily_reset = st.number_input(
            "Daily Reset (UTC Hour)",
            min_value=0,
            max_value=23,
            value=int(rules.daily_reset),
            step=1,
            key="rl_daily_reset",
        )
    with c3:
        max_loss = st.number_input(
            "Max Loss Limit (%)",
            min_value=1.0,
            max_value=30.0,
            value=float(rules.max_loss_pct),
            step=0.5,
            key="rl_max_loss",
        )
        max_type = st.selectbox(
            "Max Loss Type",
            options=["static", "trailing_closed", "trailing_equity"],
            index=["static", "trailing_closed", "trailing_equity"].index(rules.max_loss_type),
            key="rl_max_type",
        )
        trail_lock = st.checkbox(
            "Trailing floor locks at initial balance",
            value=bool(rules.trailing_lock_at_initial),
            key="rl_trail_lock",
            disabled=(max_type == "static"),
        )
        consistency = st.number_input(
            "Consistency Cap (%, 0=off)",
            min_value=0.0,
            max_value=100.0,
            value=float(rules.consistency_pct or 0.0),
            step=5.0,
            key="rl_consistency",
        )
        fee = st.number_input(
            "Challenge Fee (USD)",
            min_value=0.0,
            max_value=10000.0,
            value=float(rules.fee),
            step=50.0,
            key="rl_fee",
        )

    phases_list: list[PhaseRule] = []
    if n_phases >= 1:
        phases_list.append(PhaseRule(name="Phase 1", target_pct=p1_target, min_days=int(p1_min_days)))
    if n_phases >= 2:
        phases_list.append(PhaseRule(name="Phase 2", target_pct=p2_target, min_days=int(p2_min_days)))

    built = PropRules(
        name=r_name or "Custom Rules",
        phases=phases_list,
        daily_loss_pct=daily_loss,
        daily_basis=rules.daily_basis,
        daily_reset=int(daily_reset),
        max_loss_pct=max_loss,
        max_loss_type=max_type,  # type: ignore[arg-type]
        trailing_lock_at_initial=bool(trail_lock) and max_type != "static",
        consistency_pct=consistency if consistency > 0 else None,
        fee=fee,
        profit_split=rules.profit_split,
        payout_frequency_days=rules.payout_frequency_days,
        account_size=rules.account_size,
        max_risk_per_trade_pct=rules.max_risk_per_trade_pct,
        ambiguities=[],
    )

    built_yaml = built.to_yaml()
    if st.session_state.get("rl_last_form_yaml") != built_yaml:
        st.session_state["rl_last_form_yaml"] = built_yaml
        st.session_state["rl_yaml_editor"] = built_yaml

    edited_yaml = st.text_area(
        "YAML Specification (editable)",
        height=260,
        key="rl_yaml_editor",
    )

    b1, b2 = st.columns(2)
    with b1:
        if st.button("Apply Custom YAML to Session", key="rl_apply_manual_btn"):
            try:
                validated = parse_rules_yaml_text(edited_yaml)
                st.session_state["pp_custom_rules_yaml"] = validated.to_yaml()
                st.session_state["pp_pending_rules_mode"] = "Custom (Rules Lab)"
                st.success(f"Custom rules '{validated.name}' saved to session!")
            except Exception as exc:  # noqa: BLE001
                st.error(f"Invalid YAML / PropRules schema: {exc}")
    with b2:
        st.download_button(
            "Download Rules YAML",
            data=edited_yaml,
            file_name="proppilot_rules.yaml",
            mime="text/yaml",
            key="rl_download_yaml",
        )

render_suite_footer()
