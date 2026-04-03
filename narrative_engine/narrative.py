import random
from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass(frozen=True)
class Narrative:
	"""Structured narrative with multiple signal groups and derived features."""

	step: int
	regime: str
	components: Dict[str, Dict[str, float]]
	features: Dict[str, float]


class NarrativeEngine:
	"""Minimal structured narrative generator.

	- Narrative is multi-dimensional, not scalar sentiment.
	- `components` holds raw signals (macro/tech/rates/sentiment).
	- `features` provides agent-facing signals used for heterogeneous interpretation.
	- Signals are coupled through latent states (not independent random draws).
	"""

	REGIME_COMPONENTS = {
		"risk_on": {
			"macro": {"growth": 0.70, "inflation": 0.20},
			"tech": {"adoption": 0.60, "productivity": 0.40},
			"rates": {"policy_rate": -0.30, "credit_spread": -0.20},
			"sentiment": {"news": 0.45, "positioning": 0.20},
		},
		"tightening": {
			"macro": {"growth": 0.10, "inflation": 0.55},
			"tech": {"adoption": 0.20, "productivity": 0.10},
			"rates": {"policy_rate": 0.65, "credit_spread": 0.30},
			"sentiment": {"news": -0.25, "positioning": -0.15},
		},
		"stress": {
			"macro": {"growth": -0.55, "inflation": 0.35},
			"tech": {"adoption": -0.20, "productivity": -0.30},
			"rates": {"policy_rate": 0.20, "credit_spread": 0.85},
			"sentiment": {"news": -0.70, "positioning": -0.50},
		},
		"recovery": {
			"macro": {"growth": 0.45, "inflation": 0.05},
			"tech": {"adoption": 0.35, "productivity": 0.30},
			"rates": {"policy_rate": -0.10, "credit_spread": -0.05},
			"sentiment": {"news": 0.30, "positioning": 0.10},
		},
	}

	FEATURE_KEYS: Tuple[str, ...] = (
		"growth",
		"risk",
		"policy",
		"trend",
		"liquidity",
		"valuation_gap",
	)

	def __init__(self, regime_length: int = 10, noise_scale: float = 0.02, seed: int | None = None):
		if regime_length <= 0:
			raise ValueError("regime_length must be > 0")
		if noise_scale < 0:
			raise ValueError("noise_scale must be >= 0")

		self.regime_length = int(regime_length)
		self.noise_scale = float(noise_scale)
		self._rng = random.Random(seed)
		self._regimes = tuple(self.REGIME_COMPONENTS.keys())
		self._state = {
			"macro": 0.0,
			"risk": 0.0,
			"policy": 0.0,
			"momentum": 0.0,
			"tension": 0.0,
		}
		self._positioning_memory = 0.0

	def _clip(self, x: float, lo: float = -1.0, hi: float = 1.0) -> float:
		return max(lo, min(hi, x))

	def _update_latent_state(self, base: Dict[str, Dict[str, float]]) -> None:
		"""AR(1)-style latent state update with shared shocks.

		This creates correlation across narrative dimensions and over time.
		"""
		shared = self._rng.gauss(0.0, self.noise_scale)
		macro_shock = 0.8 * shared + self._rng.gauss(0.0, self.noise_scale * 0.4)
		risk_shock = 0.7 * shared + self._rng.gauss(0.0, self.noise_scale * 0.5)
		policy_shock = -0.3 * shared + self._rng.gauss(0.0, self.noise_scale * 0.4)
		momentum_shock = 0.4 * shared + self._rng.gauss(0.0, self.noise_scale * 0.5)
		tension_shock = -0.2 * shared + self._rng.gauss(0.0, self.noise_scale * 0.7)

		anchor_macro = 0.7 * base["macro"]["growth"] - 0.3 * base["macro"]["inflation"]
		anchor_risk = 0.8 * base["rates"]["credit_spread"] - 0.4 * base["sentiment"]["news"]
		anchor_policy = base["rates"]["policy_rate"]
		anchor_momentum = 0.6 * base["tech"]["adoption"] + 0.4 * base["sentiment"]["news"]
		# Tension encodes internal narrative contradiction (risk-on growth with fragile funding/risk backdrop).
		anchor_tension = 0.5 * base["macro"]["growth"] + 0.4 * base["tech"]["adoption"]

		self._state["macro"] = self._clip(0.85 * self._state["macro"] + 0.15 * anchor_macro + macro_shock)
		self._state["risk"] = self._clip(0.85 * self._state["risk"] + 0.15 * anchor_risk + risk_shock)
		self._state["policy"] = self._clip(0.88 * self._state["policy"] + 0.12 * anchor_policy + policy_shock)
		self._state["momentum"] = self._clip(0.80 * self._state["momentum"] + 0.20 * anchor_momentum + momentum_shock)
		self._state["tension"] = self._clip(0.90 * self._state["tension"] + 0.10 * anchor_tension + tension_shock)

	def _build_components(self, base: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
		"""Map latent state + regime anchor into structured narrative components."""
		macro_growth = self._clip(
			0.60 * base["macro"]["growth"]
			+ 0.40 * self._state["macro"]
			- 0.20 * self._state["risk"]
			+ 0.10 * self._state["tension"]
		)
		inflation = self._clip(
			0.60 * base["macro"]["inflation"]
			+ 0.30 * self._state["risk"]
			- 0.20 * self._state["macro"]
			+ 0.10 * self._state["tension"]
		)

		adoption = self._clip(
			0.60 * base["tech"]["adoption"]
			+ 0.35 * self._state["momentum"]
			- 0.15 * self._state["policy"]
			+ 0.10 * self._state["tension"]
		)
		productivity = self._clip(0.70 * base["tech"]["productivity"] + 0.30 * adoption)

		policy_rate = self._clip(
			0.65 * base["rates"]["policy_rate"]
			+ 0.35 * self._state["policy"]
			+ 0.20 * inflation
			+ 0.08 * self._state["tension"]
		)
		credit_spread = self._clip(
			0.60 * base["rates"]["credit_spread"]
			+ 0.40 * self._state["risk"]
			- 0.20 * macro_growth
			+ 0.25 * self._state["tension"]
		)

		news = self._clip(
			0.60 * base["sentiment"]["news"]
			+ 0.30 * macro_growth
			- 0.40 * credit_spread
			+ 0.20 * self._state["momentum"]
			- 0.15 * self._state["tension"]
		)
		self._positioning_memory = self._clip(0.80 * self._positioning_memory + 0.20 * news)
		positioning = self._clip(0.60 * base["sentiment"]["positioning"] + 0.40 * self._positioning_memory)

		return {
			"macro": {"growth": macro_growth, "inflation": inflation},
			"tech": {"adoption": adoption, "productivity": productivity},
			"rates": {"policy_rate": policy_rate, "credit_spread": credit_spread},
			"sentiment": {"news": news, "positioning": positioning},
		}

	def _to_features(self, components: Dict[str, Dict[str, float]]) -> Dict[str, float]:
		macro = components["macro"]
		tech = components["tech"]
		rates = components["rates"]
		sent = components["sentiment"]

		features = {
			"growth": 0.55 * macro["growth"] + 0.45 * tech["adoption"],
			"risk": 0.60 * rates["credit_spread"] + 0.40 * macro["inflation"] - 0.50 * sent["news"],
			"policy": -rates["policy_rate"],
			"trend": 0.50 * tech["productivity"] + 0.50 * sent["news"],
			"liquidity": -0.70 * rates["policy_rate"] - 0.50 * rates["credit_spread"] + 0.20 * sent["positioning"],
			"valuation_gap": 0.50 * sent["positioning"] - 0.30 * macro["growth"] + 0.30 * rates["policy_rate"],
		}
		for key in self.FEATURE_KEYS:
			features[key] = self._clip(float(features[key]))
		return features

	def generate(self, step: int) -> Narrative:
		if step < 0:
			raise ValueError("step must be >= 0")

		regime_idx = (step // self.regime_length) % len(self._regimes)
		regime = self._regimes[regime_idx]
		base = self.REGIME_COMPONENTS[regime]
		self._update_latent_state(base)
		components = self._build_components(base)
		features = self._to_features(components)
		return Narrative(step=step, regime=regime, components=components, features=features)


def narrative_features_for_step(step: int, regime_length: int = 10, noise_scale: float = 0.02, seed: int | None = None) -> Dict[str, float]:
	"""Convenience function for agents that consume feature dicts only."""
	engine = NarrativeEngine(regime_length=regime_length, noise_scale=noise_scale, seed=seed)
	return engine.generate(step).features


def example_narrative_output() -> Dict[str, object]:
	"""Example output structure."""
	n = NarrativeEngine(seed=1).generate(step=0)
	return {
		"step": n.step,
		"regime": n.regime,
		"components": n.components,
		"features": n.features,
	}
