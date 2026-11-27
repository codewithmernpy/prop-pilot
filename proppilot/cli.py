"""Command-line interface for PropPilot."""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path
from typing import Sequence

import numpy as np

from proppilot.ai import (
    PlanSession,
    RuleExtractionError,
    explain_plan,
    extract_rules_from_text,
    extract_rules_offline,
    run_copilot_query,
)
from proppilot.ingest import load_trades
from proppilot.optimize import DEFAULT_MAX_LOSE_FUNDED, OptimizationResult, run_honest_optimisation
from proppilot.policy import SizingPolicy, default_policy_for_rules
from proppilot.report import render_reports
from proppilot.rules import load_rules
from proppilot.sim import (
    BatchChallengeSummary,
    format_days,
    format_expected_attempts,
    format_expected_cost,
    format_pct,
    format_usd,
    simulate_bootstrap_challenge,
    simulate_funded_bootstrap,
    simulate_rolling_starts,
)
from proppilot.synthetic import write_sample_trades


def _resolve_default_sample() -> Path:
    cwd_path = Path("examples/trades_sample.csv")
    if cwd_path.exists():
        return cwd_path
    pkg_rel = Path(__file__).resolve().parent.parent / "examples" / "trades_sample.csv"
    if pkg_rel.exists():
        return pkg_rel
    return write_sample_trades(cwd_path)


def _format_row(label: str, sm: BatchChallengeSummary) -> str:
    med = format_days(sm.days_median)
    att = format_expected_attempts(sm.expected_attempts)
    fee = format_expected_cost(sm.expected_fees)
    return (
        f"  {label:<28} | finished={sm.n_finished:<4} unfin={sm.n_unfinished:<4} | "
        f"pass={format_pct(sm.pass_rate):>6} fail={format_pct(sm.fail_rate):>6} rest={format_pct(sm.restart_rate):>6} | "
        f"med={med:<7} att={att:<5} cost={fee}"
    )


def _cmd_simulate(args: argparse.Namespace) -> int:
    ds = load_trades(args.trades)
    rules = load_rules(args.rules, account_size=args.account_size)
    policy = SizingPolicy.from_yaml(args.policy) if args.policy else default_policy_for_rules(rules)

    roll = simulate_rolling_starts(ds, rules, policy)
    boot = simulate_bootstrap_challenge(ds, rules, policy, n_paths=args.bootstrap, seed=args.seed)
    funded = simulate_funded_bootstrap(ds, rules, policy, n_paths=args.bootstrap, seed=args.seed + 1)

    print(f"Rules: {rules.name} (${rules.account_size:,.0f}, fee ${rules.fee:,.0f})")
    print(f"Policy: {policy.label()}")
    print(_format_row("Rolling starts (all)", roll))
    print(_format_row(f"Block bootstrap ({args.bootstrap}x1y)", boot))
    print(
        f"  Funded (12m, {args.bootstrap} paths, {policy.funded_label()}) | "
        f"P(lose account)={format_pct(funded.p_lose_account):>6} | "
        f"mean payout={format_usd(funded.monthly_payout_mean_usd, '/mo')} "
        f"({f'{funded.monthly_payout_mean_pct:.2f}%/mo' if math.isfinite(funded.monthly_payout_mean_pct) else 'N/A'}, "
        f"lost accounts included) | median={format_usd(funded.monthly_payout_median_usd, '/mo')}"
    )
    for note in rules.approximation_notes:
        print(f"  [note] {note}")

    if args.json_out:
        out_p = Path(args.json_out)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(
            json.dumps(
                {
                    "rules": rules.model_dump(mode="json"),
                    "policy": policy.model_dump(mode="json"),
                    "rolling": roll.to_dict(),
                    "bootstrap": boot.to_dict(),
                    "funded": funded.to_dict(),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"Wrote JSON summary: {out_p}")
    return 0


def _cmd_optimize(args: argparse.Namespace) -> int:
    ds = load_trades(args.trades)
    rules = load_rules(args.rules, account_size=args.account_size)
    res = run_honest_optimisation(
        dataset=ds,
        rules=rules,
        objective=args.objective,
        max_fail=args.max_fail,
        n_bootstrap=args.bootstrap,
        seed=args.seed,
        max_lose_funded=args.max_lose_funded,
    )

    print(
        f"=== PropPilot Honest Optimisation ({res.objective}, max_fail={res.max_fail:.2f}, "
        f"max_lose_funded={res.max_lose_funded:.2f}) ==="
    )
    print(f"Rules: {rules.name} (${rules.account_size:,.0f}, fee ${rules.fee:,.0f})")
    print(f"Recommended Policy: {res.best_policy.label()}")
    print("Comparison (Optimise 60% vs Validate 40% vs Bootstrap):")
    print(_format_row(f"Optimise (first 60%, n={res.opt_trade_count})", res.opt_summary))
    print(_format_row(f"Validate (last 40%,  n={res.val_trade_count})", res.val_summary))
    print(_format_row(f"Bootstrap ({res.boot_summary.n_total}x1y)", res.boot_summary))
    _print_funded(res)
    print(f"PLAN CARD: {res.plan_card_summary()}")
    for w in res.warnings:
        print(f"[WARNING] {w}")

    if args.json_out:
        out_p = Path(args.json_out)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(json.dumps(res.to_dict(), indent=2), encoding="utf-8")
        print(f"Wrote JSON: {out_p}")
    return 0


def _cmd_plan(args: argparse.Namespace) -> int:
    ds = load_trades(args.trades)
    rules = load_rules(args.rules, account_size=args.account_size)
    res = run_honest_optimisation(
        dataset=ds,
        rules=rules,
        objective=args.objective,
        max_fail=args.max_fail,
        n_bootstrap=args.bootstrap,
        seed=args.seed,
        max_lose_funded=args.max_lose_funded,
    )
    session = PlanSession(
        dataset=ds,
        rules=rules,
        objective=args.objective,
        max_fail=args.max_fail,
        n_bootstrap=args.bootstrap,
        seed=args.seed,
        max_lose_funded=args.max_lose_funded,
        opt_result=res,
    )
    reply = explain_plan(session, use_ai=bool(args.ai))
    stem = args.name or f"{rules.name.lower().replace(' ', '_')}_{args.objective}"
    html_path, json_path = render_reports(
        dataset=ds,
        result=res,
        explanation_text=reply.text,
        used_ai=reply.used_ai,
        out_dir=args.out_dir,
        stem=stem,
    )
    print(f"PLAN CARD: {res.plan_card_summary()}")
    print(_format_row("Optimise (first 60%)", res.opt_summary))
    print(_format_row("Validate (last 40%)", res.val_summary))
    print(_format_row(f"Bootstrap ({res.boot_summary.n_total}x1y)", res.boot_summary))
    _print_funded(res)
    for w in res.warnings:
        print(f"[WARNING] {w}")
    if reply.fallback_reason:
        print(f"[INFO] {reply.fallback_reason}")
    print(f"HTML report: {html_path}")
    print(f"JSON report: {json_path}")
    return 0


def _cmd_rules(args: argparse.Namespace) -> int:
    src = Path(args.from_file)
    if not src.exists():
        print(f"Error: rules text file not found: {src}", file=sys.stderr)
        return 1
    text = src.read_text(encoding="utf-8")
    if not text.strip():
        print(f"Error: rules text file is empty: {src}", file=sys.stderr)
        return 1
    try:
        parsed = extract_rules_from_text(text)
        print("Extracted PropRules YAML (Claude structured output):")
    except RuleExtractionError as exc:
        print(f"[INFO] AI extraction unavailable ({exc}); using the offline keyword parser.")
        parsed = extract_rules_offline(text)
        print("Extracted PropRules YAML (offline keyword parser - review every field):")
    print(parsed.to_yaml())
    if parsed.ambiguities:
        print("AMBIGUITIES TO REVIEW & CONFIRM:")
        for idx, item in enumerate(parsed.ambiguities, 1):
            print(f"  {idx}. {item}")
    else:
        print("No ambiguities flagged.")

    if args.out:
        out_path = parsed.save_yaml(args.out)
        print(f"Saved rules YAML to {out_path}")
    return 0


def _cmd_chat(args: argparse.Namespace) -> int:
    trades_path = Path(args.trades) if args.trades else _resolve_default_sample()
    ds = load_trades(trades_path)
    rules = load_rules(args.rules, account_size=args.account_size)
    session = PlanSession(
        dataset=ds,
        rules=rules,
        objective=args.objective,
        max_fail=args.max_fail,
        n_bootstrap=args.bootstrap,
        seed=args.seed,
        max_lose_funded=args.max_lose_funded,
    )
    session.ensure_optimised()

    if args.prompt:
        reply = run_copilot_query(session, prompt=args.prompt)
        if reply.fallback_reason:
            print(f"[INFO] {reply.fallback_reason}")
        print(reply.text)
        return 0

    print("PropPilot Copilot interactive chat (type 'exit' or 'quit' to stop).")
    history: list[dict[str, str]] = []
    while True:
        try:
            user_text = input("\nYou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting chat.")
            break
        if not user_text:
            continue
        if user_text.lower() in {"exit", "quit", "q"}:
            break
        reply = run_copilot_query(session, prompt=user_text, history=history)
        if reply.fallback_reason:
            print(f"[INFO] {reply.fallback_reason}")
        print(f"\nPropPilot> {reply.text}")
        history.append({"role": "user", "content": user_text})
        history.append({"role": "assistant", "content": reply.text})
    return 0


DEMO_PLANS = (("two_step_classic", "demo_plan"), ("one_step", "demo_plan_one_step"))
DEMO_MAX_FAIL = 0.40


def _cmd_demo(args: argparse.Namespace) -> int:
    sample_path = _resolve_default_sample()
    ds = load_trades(sample_path)
    print(f"Trades: {sample_path} (SYNTHETIC sample, {len(ds):,} trades)")
    for preset, stem in DEMO_PLANS:
        rules = load_rules(preset, account_size=args.account_size)
        res = run_honest_optimisation(
            dataset=ds,
            rules=rules,
            objective="fastest",
            max_fail=DEMO_MAX_FAIL,
            n_bootstrap=args.bootstrap,
            seed=42,
            max_lose_funded=DEFAULT_MAX_LOSE_FUNDED,
        )
        session = PlanSession(
            dataset=ds,
            rules=rules,
            objective="fastest",
            max_fail=DEMO_MAX_FAIL,
            n_bootstrap=args.bootstrap,
            seed=42,
            opt_result=res,
        )
        reply = explain_plan(session, use_ai=False)
        html_path, json_path = render_reports(
            dataset=ds,
            result=res,
            explanation_text=reply.text,
            used_ai=reply.used_ai,
            out_dir=args.out_dir,
            stem=stem,
        )
        print(f"\n=== {rules.name} (objective fastest, max_fail {DEMO_MAX_FAIL:.2f}) ===")
        print(f"PLAN CARD: {res.plan_card_summary()}")
        print(_format_row("Optimise (first 60%)", res.opt_summary))
        print(_format_row("Validate (last 40%)", res.val_summary))
        print(_format_row(f"Bootstrap ({res.boot_summary.n_total}x1y)", res.boot_summary))
        _print_funded(res)
        for w in res.warnings:
            print(f"[WARNING] {w}")
        print(f"HTML report: {html_path}")
        print(f"JSON report: {json_path}")
    print("\n=== PropPilot Demo Complete ===")
    return 0


def _print_funded(res: OptimizationResult) -> None:
    fn = res.funded_summary
    print(f"FUNDED RISK: {res.funded_risk_line()}")
    mean_pct = f"{fn.monthly_payout_mean_pct:.2f}%/mo" if math.isfinite(fn.monthly_payout_mean_pct) else "N/A"
    print(
        f"Funded 12m projection ({res.best_policy.funded_label()}): mean {format_usd(fn.monthly_payout_mean_usd, '/mo')} "
        f"({mean_pct}, lost accounts included), "
        f"median {format_usd(fn.monthly_payout_median_usd, '/mo')}"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="proppilot",
        description="PropPilot: Turn any crypto strategy into a prop-firm pass plan.",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")
    sub = parser.add_subparsers(dest="command", required=True)

    # simulate
    p_sim = sub.add_parser("simulate", help="Simulate a single sizing policy.")
    p_sim.add_argument("--trades", required=True, help="Path to trades CSV or JSON.")
    p_sim.add_argument("--rules", required=True, help="Preset name or path to rules YAML.")
    p_sim.add_argument("--policy", default=None, help="Optional policy YAML path.")
    p_sim.add_argument("--account-size", type=float, default=None, help="Override account size USD.")
    p_sim.add_argument("--bootstrap", type=int, default=2000, help="Bootstrap paths (default 2000).")
    p_sim.add_argument("--seed", type=int, default=42, help="RNG seed.")
    p_sim.add_argument("--json-out", default=None, help="Optional JSON output path.")
    p_sim.set_defaults(func=_cmd_simulate)

    # optimize
    p_opt = sub.add_parser("optimize", help="Optimise on first 60% and validate on last 40%.")
    p_opt.add_argument("--trades", required=True, help="Path to trades CSV or JSON.")
    p_opt.add_argument("--rules", required=True, help="Preset name or path to rules YAML.")
    p_opt.add_argument(
        "--objective",
        choices=["fastest", "cheapest", "max_value"],
        default="fastest",
        help="Optimisation objective.",
    )
    p_opt.add_argument("--max-fail", type=float, default=0.20, help="Max non-pass fraction constraint.")
    p_opt.add_argument(
        "--max-lose-funded",
        type=float,
        default=DEFAULT_MAX_LOSE_FUNDED,
        help="Max P(lose funded account within 12m) allowed for the funded plan (default 0.15).",
    )
    p_opt.add_argument("--bootstrap", type=int, default=2000, help="Bootstrap paths (default 2000).")
    p_opt.add_argument("--account-size", type=float, default=None, help="Override account size USD.")
    p_opt.add_argument("--seed", type=int, default=42, help="RNG seed.")
    p_opt.add_argument("--json-out", default=None, help="Optional JSON output path.")
    p_opt.set_defaults(func=_cmd_optimize)

    # plan
    p_plan = sub.add_parser("plan", help="Optimise, validate, project funded, and render HTML+JSON.")
    p_plan.add_argument("--trades", required=True, help="Path to trades CSV or JSON.")
    p_plan.add_argument("--rules", required=True, help="Preset name or path to rules YAML.")
    p_plan.add_argument(
        "--objective",
        choices=["fastest", "cheapest", "max_value"],
        default="fastest",
        help="Optimisation objective.",
    )
    p_plan.add_argument("--max-fail", type=float, default=0.20, help="Max non-pass fraction constraint.")
    p_plan.add_argument(
        "--max-lose-funded",
        type=float,
        default=DEFAULT_MAX_LOSE_FUNDED,
        help="Max P(lose funded account within 12m) allowed for the funded plan (default 0.15).",
    )
    p_plan.add_argument("--bootstrap", type=int, default=2000, help="Bootstrap paths (default 2000).")
    p_plan.add_argument("--account-size", type=float, default=None, help="Override account size USD.")
    p_plan.add_argument("--out-dir", default="reports", help="Output directory for HTML & JSON.")
    p_plan.add_argument("--name", default=None, help="Report file stem.")
    p_plan.add_argument("--ai", action="store_true", help="Use Claude AI Copilot explanation if key is set.")
    p_plan.add_argument("--seed", type=int, default=42, help="RNG seed.")
    p_plan.set_defaults(func=_cmd_plan)

    # rules
    p_rules = sub.add_parser(
        "rules",
        help="Extract PropRules YAML from rules text (Claude, or the offline keyword parser without a key).",
    )
    p_rules.add_argument("--from", dest="from_file", required=True, help="Text file containing rules.")
    p_rules.add_argument("--out", default=None, help="Optional output YAML file path.")
    p_rules.set_defaults(func=_cmd_rules)

    # chat
    p_chat = sub.add_parser("chat", help="Interactive AI/offline Copilot chat.")
    p_chat.add_argument("--trades", default=None, help="Trades CSV/JSON (defaults to sample).")
    p_chat.add_argument("--rules", default="two_step_classic", help="Preset name or rules YAML.")
    p_chat.add_argument("--objective", choices=["fastest", "cheapest", "max_value"], default="fastest")
    p_chat.add_argument("--max-fail", type=float, default=0.20)
    p_chat.add_argument(
        "--max-lose-funded",
        type=float,
        default=DEFAULT_MAX_LOSE_FUNDED,
        help="Max P(lose funded account within 12m) allowed for the funded plan (default 0.15).",
    )
    p_chat.add_argument("--bootstrap", type=int, default=2000)
    p_chat.add_argument("--account-size", type=float, default=None)
    p_chat.add_argument("--seed", type=int, default=42)
    p_chat.add_argument("--prompt", default=None, help="Single question non-interactive mode.")
    p_chat.set_defaults(func=_cmd_chat)

    # demo
    p_demo = sub.add_parser("demo", help="Run full offline demo on sample trades and write reports/.")
    p_demo.add_argument("--out-dir", default="reports", help="Output directory for HTML & JSON.")
    p_demo.add_argument("--account-size", type=float, default=None, help="Optional account size USD.")
    p_demo.add_argument("--bootstrap", type=int, default=2000, help="Bootstrap paths (default 2000).")
    p_demo.set_defaults(func=_cmd_demo)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        # Never crash on characters the (redirected) console encoding cannot represent.
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
