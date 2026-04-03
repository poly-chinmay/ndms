import random
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
	sys.path.append(str(ROOT))

from agent_engine.agent import Agent
from market_engine.order_book import PriceMechanism
from narrative_engine.narrative import NarrativeEngine


def _build_agents(seed: int) -> list[Agent]:
	rng = random.Random(seed)
	agents: list[Agent] = []
	for idx, agent_type in enumerate(("trend_follower", "contrarian", "risk_averse", "risk_seeking")):
		for _ in range(2):
			agents.append(
				Agent(
					agent_type,
					capital=10_000 + rng.randint(0, 4_000),
					noise_scale=0.08 + 0.06 * rng.random(),
					rng_seed=1_000 + idx * 100 + len(agents),
				)
			)
	return agents


def evaluate_price_series(prices: list[float], lag: int = 1) -> dict[str, float | np.ndarray]:
	"""Minimal NumPy-based diagnostics for simulated price paths."""
	arr = np.asarray(prices, dtype=float)
	if arr.size < 2:
		return {
			"returns": np.array([], dtype=float),
			"kurtosis": float("nan"),
			"autocorr_returns": float("nan"),
			"autocorr_abs_returns": float("nan"),
		}

	returns = np.log(arr[1:] / arr[:-1])

	def _autocorr(x: np.ndarray, k: int) -> float:
		if x.size <= k:
			return float("nan")
		a = x[:-k]
		b = x[k:]
		a_std = np.std(a)
		b_std = np.std(b)
		if a_std == 0.0 or b_std == 0.0:
			return float("nan")
		return float(np.corrcoef(a, b)[0, 1])

	mu = np.mean(returns)
	sigma = np.std(returns)
	if sigma == 0.0:
		kurtosis = float("nan")
	else:
		kurtosis = float(np.mean(((returns - mu) / sigma) ** 4))

	return {
		"returns": returns,
		"kurtosis": kurtosis,
		"autocorr_returns": _autocorr(returns, lag),
		"autocorr_abs_returns": _autocorr(np.abs(returns), lag),
	}


def _random_shock_features(rng: random.Random) -> dict[str, float]:
	"""Baseline structured features from random shocks (no narrative engine state)."""
	features: dict[str, float] = {}
	for key in NarrativeEngine.FEATURE_KEYS:
		features[key] = max(-1.0, min(1.0, rng.gauss(0.0, 0.45)))
	return features


def run_simulation(
	steps: int = 60,
	sim_seed: int = 7,
	mode: str = "narrative",
	verbose: bool = True,
) -> list[float]:
	if mode not in {"narrative", "random_shock"}:
		raise ValueError("mode must be 'narrative' or 'random_shock'")

	narratives = NarrativeEngine(regime_length=10, noise_scale=0.02, seed=sim_seed)
	shock_rng = random.Random(sim_seed + 10_000)
	market = PriceMechanism(initial_price=100.0, impact_lambda=0.08)
	agents = _build_agents(seed=sim_seed)

	price_history: list[float] = [market.price]
	if verbose:
		print(f"Mode: {mode}")
		print(f"{'Step':>6} {'Price':>10} {'Net Flow':>10}")
		print("-" * 30)

	for step in range(steps):
		if mode == "narrative":
			features = narratives.generate(step).features
		else:
			features = _random_shock_features(shock_rng)
		current_price = market.price
		net_flow = 0

		for agent in agents:
			# 1) mark current P&L
			agent.update_pnl(current_price)

			# 2) interpret narrative and update belief
			signal = agent.interpret_narrative(features)
			agent.update_belief(features, interpreted_signal=signal)

			# 3) forced liquidation under stress, otherwise normal decision
			qty = agent.forced_sell_qty(current_price)
			if qty == 0:
				qty = agent.decide_trade(price=current_price, max_trade_size=8)

			# 4) execute with hard constraints and aggregate actual flow
			executed_qty = agent.executable_qty(qty=qty, price=current_price)

			if executed_qty != 0:
				agent.apply_fill(qty=executed_qty, price=current_price)
				net_flow += executed_qty

		# 5) market impact from aggregate executed flow
		new_price = market.update_price(net_flow)

		# 6) end-of-step P&L mark
		for agent in agents:
			agent.update_pnl(new_price)

		price_history.append(new_price)
		if verbose:
			print(f"{step:6d} {new_price:10.3f} {net_flow:10d}")

	return price_history


def run_baseline_simulation(steps: int = 60, sim_seed: int = 7) -> list[float]:
	"""Baseline: replace narrative process with random feature shocks."""
	return run_simulation(steps=steps, sim_seed=sim_seed, mode="random_shock")


def compare_ndms_vs_baseline(steps: int = 60, sim_seed: int = 7) -> dict[str, float | str]:
	"""Run both modes and print a minimal comparison."""
	ndms_prices = run_simulation(steps=steps, sim_seed=sim_seed, mode="narrative", verbose=False)
	base_prices = run_simulation(steps=steps, sim_seed=sim_seed, mode="random_shock", verbose=False)

	ndms = evaluate_price_series(ndms_prices, lag=1)
	base = evaluate_price_series(base_prices, lag=1)

	ndms_ret_std = float(np.std(ndms["returns"])) if ndms["returns"].size else float("nan")
	base_ret_std = float(np.std(base["returns"])) if base["returns"].size else float("nan")

	kurt_diff = float(ndms["kurtosis"] - base["kurtosis"])
	abs_acf_diff = float(ndms["autocorr_abs_returns"] - base["autocorr_abs_returns"])
	ret_acf_diff = float(ndms["autocorr_returns"] - base["autocorr_returns"])

	if abs_acf_diff > 0.05:
		vol_msg = "NDMS stronger"
	elif abs_acf_diff < -0.05:
		vol_msg = "Baseline stronger"
	else:
		vol_msg = "Similar"

	if abs(kurt_diff) > 0.2 or abs(ret_acf_diff) > 0.05 or abs(ndms_ret_std - base_ret_std) > 1e-4:
		dyn_msg = "Different"
	else:
		dyn_msg = "Similar"

	print("\nSimple NDMS vs Baseline Comparison")
	print(f"{'Metric':28s} {'NDMS':>10s} {'Baseline':>10s} {'Diff':>10s}")
	print("-" * 64)
	print(f"{'Kurtosis':28s} {ndms['kurtosis']:10.4f} {base['kurtosis']:10.4f} {kurt_diff:10.4f}")
	print(
		f"{'Autocorr(Returns, lag=1)':28s} "
		f"{ndms['autocorr_returns']:10.4f} {base['autocorr_returns']:10.4f} {ret_acf_diff:10.4f}"
	)
	print(
		f"{'Autocorr(|Returns|, lag=1)':28s} "
		f"{ndms['autocorr_abs_returns']:10.4f} {base['autocorr_abs_returns']:10.4f} {abs_acf_diff:10.4f}"
	)
	print(f"{'Std(Returns)':28s} {ndms_ret_std:10.4f} {base_ret_std:10.4f} {(ndms_ret_std-base_ret_std):10.4f}")

	print("\nInterpretation")
	print(f"- Return distributions different? {dyn_msg}")
	print(f"- Volatility clustering stronger? {vol_msg}")
	print(f"- Narrative system produces different dynamics? {dyn_msg}")

	return {
		"kurtosis_ndms": float(ndms["kurtosis"]),
		"kurtosis_baseline": float(base["kurtosis"]),
		"autocorr_returns_ndms": float(ndms["autocorr_returns"]),
		"autocorr_returns_baseline": float(base["autocorr_returns"]),
		"autocorr_abs_returns_ndms": float(ndms["autocorr_abs_returns"]),
		"autocorr_abs_returns_baseline": float(base["autocorr_abs_returns"]),
	}


if __name__ == "__main__":
	history = run_simulation(steps=60, sim_seed=7, mode="narrative")
	metrics = evaluate_price_series(history, lag=1)
	print("\nNarrative Metrics")
	print(f"kurtosis={metrics['kurtosis']:.4f}")
	print(f"autocorr_returns={metrics['autocorr_returns']:.4f}")
	print(f"autocorr_abs_returns={metrics['autocorr_abs_returns']:.4f}")

	print("\n" + "=" * 30)
	baseline = run_baseline_simulation(steps=60, sim_seed=7)
	baseline_metrics = evaluate_price_series(baseline, lag=1)
	print("\nBaseline Metrics (Random Shock)")
	print(f"kurtosis={baseline_metrics['kurtosis']:.4f}")
	print(f"autocorr_returns={baseline_metrics['autocorr_returns']:.4f}")
	print(f"autocorr_abs_returns={baseline_metrics['autocorr_abs_returns']:.4f}")

	compare_ndms_vs_baseline(steps=60, sim_seed=7)
