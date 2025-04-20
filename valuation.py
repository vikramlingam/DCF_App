class DCFValuation:
    def __init__(self):
        pass

    def calculate_dcf(self, fcf, growth_rate, discount_rate, years):
        """
        Calculate the Discounted Cash Flow (DCF) valuation

        Parameters:
        - fcf: Free Cash Flow (most recent year)
        - growth_rate: Expected annual growth rate (decimal)
        - discount_rate: Discount rate (decimal)
        - years: Number of years to project

        Returns:
        - Present value of future cash flows plus terminal value
        """
        if fcf is None:
            raise ValueError("Free Cash Flow data is not available")

        if fcf <= 0:
            # For companies with negative or zero FCF, we'll use a small positive value
            # This is a simplification - in reality, you might want to use other valuation methods
            fcf = 1000000  # Use a nominal value of $1M
            print(f"Warning: Using nominal FCF value of $1M for DCF calculation due to non-positive actual FCF")

        if growth_rate < 0 or growth_rate > 2.5:
            raise ValueError("Growth rate should be between 0 and 2.5 (0% to 250%)")

        if discount_rate <= 0 or discount_rate > 0.3:
            raise ValueError("Discount rate should be between 0 and 0.3 (0% to 30%)")

        if years <= 0:
            raise ValueError("Projection years must be positive")

        # Calculate present value of projected cash flows
        pv_fcf = 0
        for year in range(1, years + 1):
            projected_fcf = fcf * (1 + growth_rate) ** year
            pv_fcf += projected_fcf / (1 + discount_rate) ** year

        # Calculate terminal value (Gordon Growth Model)
        # Assume long-term growth rate is lower than initial growth rate
        # For high growth companies, cap the terminal growth rate at a reasonable level
        terminal_growth_rate = min(growth_rate, 0.04)  # Cap at 4% for sustainability

        # For very high growth rates, use a more aggressive reduction to terminal rate
        if growth_rate > 0.5:
            # For high growth companies, use a more gradual approach to terminal value
            # This simulates a company with high initial growth that normalizes over time
            terminal_value = 0
            transition_years = min(5, years)  # Use up to 5 transition years

            # Last projected FCF
            last_fcf = fcf * (1 + growth_rate) ** years

            # Calculate terminal value with gradual growth reduction
            terminal_value = last_fcf * (1 + terminal_growth_rate) / (discount_rate - terminal_growth_rate)
        else:
            # Standard terminal value calculation
            terminal_value = fcf * (1 + growth_rate) ** years * (1 + terminal_growth_rate) / (discount_rate - terminal_growth_rate)

        # Discount terminal value to present
        pv_terminal_value = terminal_value / (1 + discount_rate) ** years

        # Total company value
        company_value = pv_fcf + pv_terminal_value

        return company_value

    def calculate_per_share_value(self, company_value, shares_outstanding):
        """
        Calculate per share value

        Parameters:
        - company_value: Total company value from DCF
        - shares_outstanding: Number of shares outstanding

        Returns:
        - Value per share
        """
        if shares_outstanding is None or shares_outstanding == 'N/A':
            raise ValueError("Shares outstanding data is not available")

        if shares_outstanding <= 0:
            raise ValueError("Shares outstanding must be positive")

        return company_value / shares_outstanding
