class PriceMechanism:
    """Minimal linear-impact price process.

    Price update rule:
    ΔP = λ × net_flow
    P(t+1) = P(t) + ΔP
    """

    def __init__(self, initial_price: float, impact_lambda: float = 0.01):
        if initial_price <= 0:
            raise ValueError("initial_price must be > 0")
        if impact_lambda < 0:
            raise ValueError("impact_lambda must be >= 0")

        self.price = float(initial_price)
        self.impact_lambda = float(impact_lambda)

    def update_price(self, net_flow: float) -> float:
        """Update and return price from aggregate signed order flow."""
        delta = self.impact_lambda * float(net_flow)
        self.price = max(0.01, self.price + delta)
        return self.price
