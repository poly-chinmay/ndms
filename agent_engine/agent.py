import random
import math
from typing import Dict, Optional


class Agent:
	"""Minimal NDMS agent.

	Core ideas:
	- The input narrative is a structured object (dict of features), not a scalar sentiment.
	- Agent type controls how narrative features are weighted.
	- Interpretation includes noise to create heterogeneous beliefs.
	- Belief updates use memory via decay.
	- Trade decisions are constrained by capital and inventory.
	"""

	# Per-type narrative feature weights.
	# Keys are expected narrative fields; missing fields default to 0.0.
	TYPE_WEIGHTS = {
		"trend_follower": {
			"growth": 0.8,
			"risk": -0.4,
			"policy": 0.3,
			"trend": 1.2,
			"liquidity": 0.2,
			"valuation_gap": -0.3,
		},
		"contrarian": {
			"growth": 0.4,
			"risk": 0.5,
			"policy": -0.1,
			"trend": -1.1,
			"liquidity": 0.1,
			"valuation_gap": 0.8,
		},
		"risk_averse": {
			"growth": 0.3,
			"risk": -1.3,
			"policy": 0.2,
			"trend": 0.2,
			"liquidity": 0.4,
			"valuation_gap": 0.1,
		},
		"risk_seeking": {
			"growth": 0.7,
			"risk": 0.6,
			"policy": 0.2,
			"trend": 0.5,
			"liquidity": 0.2,
			"valuation_gap": 0.2,
		},
	}

	# Per-type behavior profile.
	# - decay: memory strength (higher => slower belief change)
	# - adapt: speed of belief adjustment toward memory-implied target
	# - threshold: minimum absolute belief required to trade
	# - risk_budget: max fraction of cash to deploy per decision
	# - max_inventory_frac: max long inventory notional as a fraction of wealth
	# - signal_gain: non-linear gain from memory signal -> target belief
	TYPE_PROFILE = {
		"trend_follower": {
			"decay": 0.92,
			"adapt": 0.14,
			"threshold": 0.06,
			"risk_budget": 0.28,
			"max_inventory_frac": 0.80,
			"signal_gain": 1.7,
			"belief_bias": 0.08,
			"can_short": False,
			"max_short_inventory_frac": 0.0,
		},
		"contrarian": {
			"decay": 0.55,
			"adapt": 0.60,
			"threshold": 0.02,
			"risk_budget": 0.26,
			"max_inventory_frac": 0.65,
			"signal_gain": 1.3,
			"belief_bias": -0.12,
			"can_short": True,
			"max_short_inventory_frac": 0.90,
		},
		"risk_averse": {
			"decay": 0.96,
			"adapt": 0.08,
			"threshold": 0.10,
			"risk_budget": 0.12,
			"max_inventory_frac": 0.40,
			"signal_gain": 0.7,
			"belief_bias": -0.04,
			"can_short": False,
			"max_short_inventory_frac": 0.0,
		},
		"risk_seeking": {
			"decay": 0.40,
			"adapt": 0.75,
			"threshold": 0.015,
			"risk_budget": 0.45,
			"max_inventory_frac": 1.00,
			"signal_gain": 2.1,
			"belief_bias": 0.02,
			"can_short": True,
			"max_short_inventory_frac": 1.20,
		},
	}

	# Minimal global risk controls (kept simple and interpretable).
	MAX_STEP_TURNOVER_FRAC = 0.15  # max traded notional per step as fraction of equity
	MIN_RISK_SCALE = 0.25  # floor so agents do not fully freeze after losses
	MIN_CASH_FRAC = 0.20  # forced-sell trigger: cash below 20% of initial cash
	MAX_DRAWDOWN = 0.35  # forced-sell trigger: drawdown above 35%
	FORCED_SELL_FRACTION = 0.30  # base forced liquidation fraction
	LOSS_LIQUIDATE_PARTIAL = 0.12  # trigger: unrealized loss >= 12% of initial capital
	LOSS_LIQUIDATE_FULL = 0.20  # trigger: unrealized loss >= 20% of initial capital
	PARTIAL_LIQUIDATION_FRACTION = 0.70  # aggressive partial exit
	LOSS_AVERSION = 3.0  # losses reduce risk-taking more strongly than gains increase it
	GAIN_SENSITIVITY = 0.4
	EXPOSURE_PENALTY_POWER = 1.5

	def __init__(
		self,
		agent_type: str,
		capital: float,
		position: int = 0,
		average_cost: float = 0.0,
		belief: float = 0.0,
		decay: Optional[float] = None,
		noise_scale: float = 0.15,
		rng_seed: Optional[int] = None,
	):
		if agent_type not in self.TYPE_WEIGHTS:
			raise ValueError(
				f"Unknown agent_type '{agent_type}'. Valid types: {list(self.TYPE_WEIGHTS)}"
			)
		if capital < 0:
			raise ValueError("capital must be >= 0")
		if position < 0 and not self.TYPE_PROFILE[agent_type]["can_short"]:
			raise ValueError("negative starting position is not allowed for this agent type")
		if average_cost < 0:
			raise ValueError("average_cost must be >= 0")
		if position != 0 and average_cost <= 0:
			raise ValueError("average_cost must be > 0 when position != 0")
		if noise_scale < 0:
			raise ValueError("noise_scale must be >= 0")

		default_decay = self.TYPE_PROFILE[agent_type]["decay"]
		if decay is None:
			decay = float(default_decay)
		if not (0.0 <= decay <= 1.0):
			raise ValueError("decay must be in [0, 1]")

		self.type = agent_type
		self.capital = float(capital)  # cash
		self.initial_capital = float(capital)
		self.position = int(position)
		self.average_cost = float(average_cost)
		self.belief = float(belief)

		self.decay = float(decay)
		self.noise_scale = float(noise_scale)
		self._rng = random.Random(rng_seed)
		self._idiosyncratic_bias = self._rng.gauss(0.0, self.noise_scale * 0.5)
		self.last_signal = 0.0
		self.signal_memory = 0.0
		initial_equity = self.capital + self.position * (self.average_cost if self.position > 0 else 0.0)
		self.peak_equity = max(1e-9, float(initial_equity))
		self.unrealized_pnl_value = 0.0
		self.equity = float(initial_equity)

	def update_pnl(self, current_price: float) -> float:
		"""Mark-to-market unrealized P&L and equity for current step."""
		self.unrealized_pnl_value = self.unrealized_pnl(current_price)
		self.equity = self.capital + self.position * current_price
		self.peak_equity = max(self.peak_equity, self.equity)
		return self.unrealized_pnl_value

	def forced_sell_qty(self, current_price: float) -> int:
		"""Return mandatory liquidation quantity when stress constraints are breached.

		Triggers:
		- cash below threshold of initial cash
		- drawdown above max drawdown
		- unrealized loss threshold breach (aggressive liquidation)

		Returns:
		- negative qty for long liquidation
		- positive qty for short cover
		"""
		if current_price <= 0:
			raise ValueError("current_price must be > 0")
		if self.position == 0:
			return 0

		equity = max(0.0, self.capital + self.position * current_price)
		self.peak_equity = max(self.peak_equity, equity)
		drawdown = max(0.0, (self.peak_equity - equity) / self.peak_equity)
		loss_ratio_initial = max(0.0, -self.unrealized_pnl(current_price)) / max(self.initial_capital, 1e-9)

		low_cash = self.capital < (self.initial_capital * self.MIN_CASH_FRAC)
		high_drawdown = drawdown > self.MAX_DRAWDOWN
		loss_partial = loss_ratio_initial >= self.LOSS_LIQUIDATE_PARTIAL
		loss_full = loss_ratio_initial >= self.LOSS_LIQUIDATE_FULL

		if not (low_cash or high_drawdown or loss_partial):
			return 0

		if loss_full:
			return -self.position

		if loss_partial:
			liq_qty = max(1, int(abs(self.position) * self.PARTIAL_LIQUIDATION_FRACTION))
			return -min(liq_qty, abs(self.position)) if self.position > 0 else min(liq_qty, abs(self.position))

		liq_qty = max(1, int(abs(self.position) * self.FORCED_SELL_FRACTION))
		return -min(liq_qty, abs(self.position)) if self.position > 0 else min(liq_qty, abs(self.position))

	def interpret_narrative(self, narrative: Dict[str, float]) -> float:
		"""Convert a structured narrative dict into a noisy private signal."""
		if not isinstance(narrative, dict):
			raise TypeError("narrative must be a dict of feature -> numeric value")

		weights = self.TYPE_WEIGHTS[self.type]
		signal = 0.0
		for feature, weight in weights.items():
			value = narrative.get(feature, 0.0)
			try:
				signal += weight * float(value)
			except (TypeError, ValueError) as exc:
				raise TypeError(f"Narrative feature '{feature}' must be numeric") from exc

		noise = self._rng.gauss(0.0, self.noise_scale)
		private_signal = signal + noise + self._idiosyncratic_bias
		self.last_signal = private_signal
		return private_signal

	def update_belief(
		self,
		narrative: Dict[str, float],
		interpreted_signal: Optional[float] = None,
	) -> float:
		"""Update belief with explicit memory decay and non-linear signal assimilation.

		This is intentionally not a direct mapping from signal to belief.
		"""
		if interpreted_signal is None:
			new_signal = self.interpret_narrative(narrative)
		else:
			new_signal = float(interpreted_signal)
			self.last_signal = new_signal
		profile = self.TYPE_PROFILE[self.type]

		# Explicit memory state of interpreted narrative.
		self.signal_memory = self.decay * self.signal_memory + (1.0 - self.decay) * new_signal

		# Belief moves toward a bounded target derived from memory, not raw signal.
		target = math.tanh(profile["signal_gain"] * self.signal_memory + profile.get("belief_bias", 0.0))
		self.belief = self.belief + profile["adapt"] * (target - self.belief)
		return self.belief

	def decide_trade(
		self,
		price: float,
		belief_threshold: Optional[float] = None,
		max_trade_size: int = 5,
	) -> int:
		"""Return signed quantity to trade (+buy, -sell, 0 hold).

		Decision depends on belief and constraints:
		- Buy only if belief > threshold and enough capital.
		- Sell only if belief < -threshold and enough inventory.
		"""
		if price <= 0:
			raise ValueError("price must be > 0")
		if max_trade_size <= 0:
			return 0
		if belief_threshold is None:
			belief_threshold = self.TYPE_PROFILE[self.type]["threshold"]

		# Confidence controls aggressiveness.
		confidence = abs(self.belief) - belief_threshold
		if confidence <= 0:
			return 0

		# Conviction in [0, 1), stronger beliefs saturate naturally.
		conviction = math.tanh(2.5 * confidence)
		profile = self.TYPE_PROFILE[self.type]
		direction = 1 if self.belief > 0 else -1

		# Use current marked-to-market equity, not only cash.
		equity = max(0.0, self.capital + self.position * price)
		if equity <= 0:
			return 0

		self.peak_equity = max(self.peak_equity, equity)
		drawdown = max(0.0, (self.peak_equity - equity) / self.peak_equity)
		risk_scale = max(self.MIN_RISK_SCALE, 1.0 - drawdown)

		# Asymmetric P&L response: losses shrink confidence more than gains boost it.
		pnl = self.unrealized_pnl(current_price=price)
		pnl_ratio = pnl / max(equity, 1e-9)
		loss_ratio = max(0.0, -pnl_ratio)
		gain_ratio = max(0.0, pnl_ratio)
		confidence_scale = 1.0 + self.GAIN_SENSITIVITY * gain_ratio - self.LOSS_AVERSION * loss_ratio
		confidence_scale = max(0.15, min(1.25, confidence_scale))
		adjusted_conviction = conviction * confidence_scale

		# Exposure sensitivity: large existing position reduces additional buying.
		exposure_ratio = min(1.0, (abs(self.position) * price) / max(equity, 1e-9))
		exposure_penalty = (1.0 - exposure_ratio) ** self.EXPOSURE_PENALTY_POWER

		# Inventory target makes behavior stable and state-dependent.
		max_inventory = int((equity * profile["max_inventory_frac"]) // price)
		max_short_inventory = int((equity * profile.get("max_short_inventory_frac", 0.0)) // price)
		# Loss amplification: underwater agents de-risk faster (stronger on losses).
		derisk_multiplier = max(0.2, 1.0 - self.LOSS_AVERSION * loss_ratio)
		if direction > 0:
			target_position = int(max_inventory * adjusted_conviction * derisk_multiplier)
		elif profile.get("can_short", False):
			target_position = -int(max_short_inventory * adjusted_conviction * derisk_multiplier)
		else:
			target_position = 0
		position_gap = target_position - self.position
		if position_gap == 0:
			return 0

		# Scale notional trade speed by equity, conviction, and drawdown risk scaling.
		trade_notional_budget = self.capital * profile["risk_budget"] * adjusted_conviction * risk_scale
		if abs(target_position) > abs(self.position):
			trade_notional_budget *= exposure_penalty
		# Hard cap to avoid unrealistic one-step growth.
		turnover_cap = equity * self.MAX_STEP_TURNOVER_FRAC
		trade_notional_budget = min(trade_notional_budget, turnover_cap)
		desired_size = min(max_trade_size, int(trade_notional_budget // price), abs(position_gap))
		if desired_size <= 0:
			return 0

		if position_gap > 0:
			affordable = int(self.capital // price)
			qty = min(desired_size, affordable)
			return qty

		if position_gap < 0:
			if profile.get("can_short", False):
				return -desired_size
			qty = min(desired_size, max(0, self.position))
			return -qty

		return 0

	def executable_qty(self, qty: int, price: float) -> int:
		"""Clamp proposed qty to executable qty under cash/inventory/short limits."""
		if price <= 0:
			raise ValueError("price must be > 0")
		if qty == 0:
			return 0

		profile = self.TYPE_PROFILE[self.type]
		if qty > 0:
			affordable = int(self.capital // price)
			return min(qty, max(0, affordable))

		sell_qty = -qty
		if not profile.get("can_short", False):
			return -min(sell_qty, max(0, self.position))

		equity = max(0.0, self.capital + self.position * price)
		max_short_inventory = int((equity * profile.get("max_short_inventory_frac", 0.0)) // price)
		current_short = max(0, -self.position)
		short_room = max(0, max_short_inventory - current_short)
		long_inventory = max(0, self.position)
		executable = min(sell_qty, long_inventory + short_room)
		return -max(0, executable)

	def apply_fill(self, qty: int, price: float) -> None:
		"""Apply an executed trade to capital and position.

		qty > 0: buy
		qty < 0: sell
		"""
		if price <= 0:
			raise ValueError("price must be > 0")
		if qty == 0:
			return

		if qty > 0:
			cost = qty * price
			if cost > self.capital:
				raise ValueError("insufficient capital for buy fill")
			self.capital -= cost

			if self.position >= 0:
				# Add/increase long.
				total_cost_basis = self.average_cost * self.position + cost
				self.position += qty
				self.average_cost = total_cost_basis / self.position
				return

			# Cover short partially/fully, or cross to long.
			short_abs = -self.position
			if qty < short_abs:
				self.position += qty
				return
			if qty == short_abs:
				self.position = 0
				self.average_cost = 0.0
				return

			# Cross from short to long.
			remaining = qty - short_abs
			self.position = remaining
			self.average_cost = price
			return

		sell_qty = -qty
		self.capital += sell_qty * price

		profile = self.TYPE_PROFILE[self.type]
		if self.position <= 0:
			if not profile.get("can_short", False):
				raise ValueError("short selling not allowed for this agent type")
			# Add/increase short.
			short_abs = -self.position
			total_short_basis = self.average_cost * short_abs + sell_qty * price
			new_short_abs = short_abs + sell_qty
			self.position = -new_short_abs
			self.average_cost = total_short_basis / new_short_abs
			return

		# Reduce long, or cross to short.
		if sell_qty < self.position:
			self.position -= sell_qty
			return
		if sell_qty == self.position:
			self.position = 0
			self.average_cost = 0.0
			return

		if not profile.get("can_short", False):
			raise ValueError("insufficient position for sell fill")
		remaining = sell_qty - self.position
		self.position = -remaining
		self.average_cost = price

	def unrealized_pnl(self, current_price: float) -> float:
		"""Unrealized P&L based on current mark and average entry cost.

		unrealized_pnl = position * (current_price - average_cost)
		"""
		if current_price <= 0:
			raise ValueError("current_price must be > 0")
		if self.position == 0:
			return 0.0
		return self.position * (current_price - self.average_cost)
