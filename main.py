import random

from agent_engine.agent import Agent
from market_engine.order_book import PriceMechanism
from narrative_engine.narrative import NarrativeEngine


def _build_agents(sim_seed: int) -> list[Agent]:
	"""Create deterministic but heterogeneous agent population."""
	rng = random.Random(sim_seed)
	agents: list[Agent] = []
	# Two agents per type with deterministic per-agent variation.
	for idx, agent_type in enumerate(("trend_follower", "contrarian", "risk_averse", "risk_seeking")):
		for _ in range(2):
			capital = 10_000 + rng.randint(0, 4_000)
			noise_scale = 0.08 + 0.06 * rng.random()
			rng_seed = 1000 + idx * 100 + len(agents)
			agents.append(
				Agent(
					agent_type,
					capital=capital,
					noise_scale=noise_scale,
					rng_seed=rng_seed,
				)
			)
	return agents


def run_simulation(steps: int = 40, sim_seed: int = 7) -> list[float]:
	# Components
	narratives = NarrativeEngine(regime_length=10, noise_scale=0.02, seed=sim_seed)
	market = PriceMechanism(initial_price=100.0, impact_lambda=0.08)

	# Same narrative, heterogeneous interpretation + risk behavior.
	agents = _build_agents(sim_seed=sim_seed)

	price_history: list[float] = [market.price]

	for step in range(steps):
		narrative = narratives.generate(step)
		current_price = market.price

		net_flow = 0
		for agent in agents:
			# Per-step mark-to-market at current market price.
			agent.update_pnl(current_price)

			# Forced liquidation under stress (cash pressure or drawdown breach).
			forced_qty = agent.forced_sell_qty(current_price)
			if forced_qty != 0:
				executed_forced = agent.executable_qty(qty=forced_qty, price=current_price)
				if executed_forced != 0:
					agent.apply_fill(qty=executed_forced, price=current_price)
					net_flow += executed_forced

			agent.update_belief(narrative.features)
			proposed_qty = agent.decide_trade(price=current_price, max_trade_size=8)

			# Aggregate executed flow (not just proposed flow).
			executed_qty = agent.executable_qty(qty=proposed_qty, price=current_price)

			if executed_qty != 0:
				agent.apply_fill(qty=executed_qty, price=current_price)
				net_flow += executed_qty

		new_price = market.update_price(net_flow)
		price_history.append(new_price)

		print(
			f"step={step:02d} regime={narrative.regime:10s} "
			f"net_flow={net_flow:+4d} price={new_price:8.3f}"
		)

	return price_history


if __name__ == "__main__":
	history = run_simulation(steps=50, sim_seed=7)
	print("\nprice_history:")
	print(history)
