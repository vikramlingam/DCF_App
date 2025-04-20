import gradio as gr
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from data_fetcher import DataFetcher
from valuation import DCFValuation
from utils import (format_number, format_metrics_table, create_price_chart, 
                  format_financial_statement, create_financial_chart, create_spider_chart,
                  prepare_financial_table, create_key_metrics_chart, create_growth_chart,
                  create_margin_chart, create_multi_year_growth_chart, create_ratio_chart)

# Initialize classes
data_fetcher = DataFetcher()
dcf_valuation = DCFValuation()

def analyze_stock(ticker, growth_rate, discount_rate, projection_years, format_type):
    # Close any existing matplotlib figures to prevent memory issues
    plt.close('all')

    try:
        # Validate inputs
        if not ticker:
            return {"error": "Please enter a ticker symbol"}, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None

        ticker = ticker.upper().strip()

        # Fetch data
        metrics = data_fetcher.get_key_metrics(ticker)
        price_history = data_fetcher.get_price_history(ticker)
        financial_statements = data_fetcher.get_financial_statements(ticker)

        # Get currency symbol
        currency_symbol = metrics.get('Currency Symbol', '$')

        # Format metrics for display
        formatted_metrics = format_metrics_table(metrics)

        # Calculate DCF valuation
        try:
            fcf = data_fetcher.get_free_cash_flow(ticker)
            shares_outstanding = metrics.get('Shares Outstanding', None)

            if fcf is not None and shares_outstanding and shares_outstanding != 'N/A':
                company_value = dcf_valuation.calculate_dcf(
                    fcf=fcf,
                    growth_rate=growth_rate/100,  # Convert percentage to decimal
                    discount_rate=discount_rate/100,  # Convert percentage to decimal
                    years=int(projection_years)
                )

                # Format valuation results
                valuation_results = {
                    "Company Value": format_number(company_value, currency_symbol=currency_symbol, format_type=format_type),
                    "Current Market Cap": formatted_metrics.get("Market Cap", "N/A"),
                    "Free Cash Flow": format_number(fcf, currency_symbol=currency_symbol, format_type=format_type),
                    "Growth Rate": f"{growth_rate:.1f}%",
                    "Discount Rate": f"{discount_rate:.1f}%",
                    "Projection Years": int(projection_years)
                }

                if shares_outstanding:
                    per_share_value = dcf_valuation.calculate_per_share_value(company_value, shares_outstanding)
                    current_price = metrics.get('Current Price', None)

                    valuation_results["Estimated Share Value"] = f"{currency_symbol}{per_share_value:.2f}"
                    valuation_results["Current Share Price"] = f"{currency_symbol}{current_price:.2f}" if current_price else "N/A"

                    if current_price:
                        upside = (per_share_value / current_price - 1) * 100
                        valuation_results["Potential Upside"] = f"{upside:.1f}%"
            else:
                valuation_results = {"error": f"Insufficient data for DCF valuation. FCF: {fcf}, Shares: {shares_outstanding}"}
        except Exception as e:
            valuation_results = {"error": f"Valuation error: {str(e)}"}

        # Create price chart
        price_fig = create_price_chart(price_history)

        # Create spider chart with enhanced metrics
        spider_fig = create_spider_chart(metrics, f"{ticker} Financial Metrics")

        # Prepare financial statements for display - Annual
        annual_income_table = prepare_financial_table(
            financial_statements['income_stmt'], 
            currency_symbol=currency_symbol,
            format_type=format_type
        )

        annual_balance_table = prepare_financial_table(
            financial_statements['balance_sheet'], 
            currency_symbol=currency_symbol,
            format_type=format_type
        )

        annual_cash_flow_table = prepare_financial_table(
            financial_statements['cash_flow'], 
            currency_symbol=currency_symbol,
            format_type=format_type
        )

        # Prepare financial statements for display - Quarterly
        quarterly_income_table = prepare_financial_table(
            financial_statements['quarterly_income_stmt'], 
            currency_symbol=currency_symbol,
            format_type=format_type
        )

        quarterly_balance_table = prepare_financial_table(
            financial_statements['quarterly_balance_sheet'], 
            currency_symbol=currency_symbol,
            format_type=format_type
        )

        quarterly_cash_flow_table = prepare_financial_table(
            financial_statements['quarterly_cash_flow'], 
            currency_symbol=currency_symbol,
            format_type=format_type
        )

        # Create financial charts with error handling
        try:
            income_fig = create_financial_chart(financial_statements['income_stmt'], 
                                               f"{ticker} Income Statement", 'bar')
        except Exception as e:
            income_fig = plt.figure(figsize=(10, 6))
            plt.text(0.5, 0.5, f"Error creating income statement chart: {str(e)}", 
                     horizontalalignment='center', verticalalignment='center', fontsize=12)
            plt.axis('off')

        try:
            balance_fig = create_financial_chart(financial_statements['balance_sheet'], 
                                                f"{ticker} Balance Sheet", 'bar')
        except Exception as e:
            balance_fig = plt.figure(figsize=(10, 6))
            plt.text(0.5, 0.5, f"Error creating balance sheet chart: {str(e)}", 
                     horizontalalignment='center', verticalalignment='center', fontsize=12)
            plt.axis('off')

        try:
            cash_flow_fig = create_financial_chart(financial_statements['cash_flow'], 
                                                  f"{ticker} Cash Flow", 'bar')
        except Exception as e:
            cash_flow_fig = plt.figure(figsize=(10, 6))
            plt.text(0.5, 0.5, f"Error creating cash flow chart: {str(e)}", 
                     horizontalalignment='center', verticalalignment='center', fontsize=12)
            plt.axis('off')

        # Create quarterly financial charts with error handling
        try:
            q_income_fig = create_financial_chart(financial_statements['quarterly_income_stmt'], 
                                                 f"{ticker} Quarterly Income Statement", 'bar')
        except Exception as e:
            q_income_fig = plt.figure(figsize=(10, 6))
            plt.text(0.5, 0.5, f"Quarterly income data not available", 
                     horizontalalignment='center', verticalalignment='center', fontsize=12)
            plt.axis('off')

        try:
            q_balance_fig = create_financial_chart(financial_statements['quarterly_balance_sheet'], 
                                                  f"{ticker} Quarterly Balance Sheet", 'bar')
        except Exception as e:
            q_balance_fig = plt.figure(figsize=(10, 6))
            plt.text(0.5, 0.5, f"Quarterly balance sheet data not available", 
                     horizontalalignment='center', verticalalignment='center', fontsize=12)
            plt.axis('off')

        try:
            q_cash_flow_fig = create_financial_chart(financial_statements['quarterly_cash_flow'], 
                                                    f"{ticker} Quarterly Cash Flow", 'bar')
        except Exception as e:
            q_cash_flow_fig = plt.figure(figsize=(10, 6))
            plt.text(0.5, 0.5, f"Quarterly cash flow data not available", 
                     horizontalalignment='center', verticalalignment='center', fontsize=12)
            plt.axis('off')

        # Create additional analysis charts with error handling
        try:
            revenue_growth_fig = create_growth_chart(
                financial_statements['income_stmt'], 
                'Total Revenue', 
                f"{ticker} Revenue Growth"
            )
        except Exception as e:
            revenue_growth_fig = plt.figure(figsize=(10, 6))
            plt.text(0.5, 0.5, f"Revenue growth data not available", 
                     horizontalalignment='center', verticalalignment='center', fontsize=12)
            plt.axis('off')

        try:
            margin_fig = create_margin_chart(
                financial_statements['income_stmt'],
                f"{ticker} Margin Analysis"
            )
        except Exception as e:
            margin_fig = plt.figure(figsize=(10, 6))
            plt.text(0.5, 0.5, f"Margin analysis data not available", 
                     horizontalalignment='center', verticalalignment='center', fontsize=12)
            plt.axis('off')

        # Create key metrics charts with error handling
        try:
            key_metrics_income = create_key_metrics_chart(
                financial_statements['income_stmt'],
                f"{ticker} Key Income Metrics",
                ['Total Revenue', 'Gross Profit', 'Operating Income', 'Net Income'],
                currency_symbol
            )
        except Exception as e:
            key_metrics_income = plt.figure(figsize=(10, 6))
            plt.text(0.5, 0.5, f"Income metrics data not available", 
                     horizontalalignment='center', verticalalignment='center', fontsize=12)
            plt.axis('off')

        try:
            key_metrics_balance = create_key_metrics_chart(
                financial_statements['balance_sheet'],
                f"{ticker} Key Balance Sheet Metrics",
                ['Total Assets', 'Total Liabilities Net Minority Interest', 'Total Equity Gross Minority Interest'],
                currency_symbol
            )
        except Exception as e:
            key_metrics_balance = plt.figure(figsize=(10, 6))
            plt.text(0.5, 0.5, f"Balance sheet metrics data not available", 
                     horizontalalignment='center', verticalalignment='center', fontsize=12)
            plt.axis('off')

        try:
            key_metrics_cash = create_key_metrics_chart(
                financial_statements['cash_flow'],
                f"{ticker} Key Cash Flow Metrics",
                ['Operating Cash Flow', 'Free Cash Flow', 'Capital Expenditures'],
                currency_symbol
            )
        except Exception as e:
            key_metrics_cash = plt.figure(figsize=(10, 6))
            plt.text(0.5, 0.5, f"Cash flow metrics data not available", 
                     horizontalalignment='center', verticalalignment='center', fontsize=12)
            plt.axis('off')

        # Create ratio charts with error handling
        try:
            profitability_fig = create_ratio_chart(
                financial_statements['income_stmt'],
                f"{ticker} Profitability Ratios",
                'profitability'
            )
        except Exception as e:
            profitability_fig = plt.figure(figsize=(10, 6))
            plt.text(0.5, 0.5, f"Profitability ratio data not available", 
                     horizontalalignment='center', verticalalignment='center', fontsize=12)
            plt.axis('off')

        try:
            efficiency_fig = create_ratio_chart(
                financial_statements['balance_sheet'],
                f"{ticker} Efficiency Ratios",
                'efficiency'
            )
        except Exception as e:
            efficiency_fig = plt.figure(figsize=(10, 6))
            plt.text(0.5, 0.5, f"Efficiency ratio data not available", 
                     horizontalalignment='center', verticalalignment='center', fontsize=12)
            plt.axis('off')

        return (formatted_metrics, valuation_results, price_fig, 
                annual_income_table, annual_balance_table, annual_cash_flow_table,
                quarterly_income_table, quarterly_balance_table, quarterly_cash_flow_table,
                income_fig, balance_fig, cash_flow_fig,
                q_income_fig, q_balance_fig, q_cash_flow_fig,
                spider_fig, revenue_growth_fig, margin_fig,
                key_metrics_income, key_metrics_balance, key_metrics_cash,
                profitability_fig)

    except Exception as e:
        error_msg = {"error": f"Error: {str(e)}"}
        # Create empty figures for all plots
        empty_fig = plt.figure(figsize=(10, 6))
        plt.text(0.5, 0.5, f"Error: {str(e)}", 
                 horizontalalignment='center', verticalalignment='center', fontsize=14)
        plt.axis('off')

        return (error_msg, {"error": str(e)}, empty_fig, 
                {"error": "Data unavailable"}, {"error": "Data unavailable"}, {"error": "Data unavailable"},
                {"error": "Data unavailable"}, {"error": "Data unavailable"}, {"error": "Data unavailable"},
                empty_fig, empty_fig, empty_fig, empty_fig, empty_fig, empty_fig, 
                empty_fig, empty_fig, empty_fig, empty_fig, empty_fig, empty_fig, empty_fig)

# Create Gradio interface
with gr.Blocks(title="Stock DCF Valuation Tool") as app:
    gr.Markdown("# Stock DCF Valuation Tool")
    gr.Markdown("Enter a stock ticker and DCF assumptions to get a valuation")

    with gr.Row():
        with gr.Column(scale=1):
            ticker_input = gr.Textbox(label="Stock Ticker (e.g., AAPL, RELIANCE.NS)", placeholder="Enter ticker...")

            with gr.Row():
                growth_rate = gr.Slider(minimum=0, maximum=250, value=15, step=1, label="Growth Rate (%)")
                discount_rate = gr.Slider(minimum=5, maximum=20, value=10, step=0.1, label="Discount Rate (%)")

            projection_years = gr.Slider(minimum=1, maximum=10, value=5, step=1, label="Projection Years")

            format_type = gr.Radio(
                ["auto", "comma", "millions", "billions"], 
                label="Number Format", 
                value="millions"
            )

            analyze_button = gr.Button("Analyze Stock", variant="primary")

        with gr.Column(scale=2):
            with gr.Tabs():
                with gr.TabItem("Key Metrics"):
                    metrics_output = gr.JSON(label="Key Metrics")
                    spider_chart = gr.Plot(label="Financial Metrics Radar")

                with gr.TabItem("DCF Valuation"):
                    valuation_output = gr.JSON(label="DCF Valuation Results")
                    price_chart = gr.Plot(label="Price History")

                with gr.TabItem("Annual Financials"):
                    with gr.Tabs():
                        with gr.TabItem("Income Statement"):
                            income_chart = gr.Plot(label="Income Statement Chart")
                            annual_income_output = gr.JSON(label="Annual Income Statement")

                        with gr.TabItem("Balance Sheet"):
                            balance_chart = gr.Plot(label="Balance Sheet Chart")
                            annual_balance_output = gr.JSON(label="Annual Balance Sheet")

                        with gr.TabItem("Cash Flow"):
                            cash_flow_chart = gr.Plot(label="Cash Flow Chart")
                            annual_cash_flow_output = gr.JSON(label="Annual Cash Flow")

                with gr.TabItem("Quarterly Financials"):
                    with gr.Tabs():
                        with gr.TabItem("Income Statement"):
                            q_income_chart = gr.Plot(label="Quarterly Income Statement Chart")
                            quarterly_income_output = gr.JSON(label="Quarterly Income Statement")

                        with gr.TabItem("Balance Sheet"):
                            q_balance_chart = gr.Plot(label="Quarterly Balance Sheet Chart")
                            quarterly_balance_output = gr.JSON(label="Quarterly Balance Sheet")

                        with gr.TabItem("Cash Flow"):
                            q_cash_flow_chart = gr.Plot(label="Quarterly Cash Flow Chart")
                            quarterly_cash_flow_output = gr.JSON(label="Quarterly Cash Flow")

                with gr.TabItem("Financial Analysis"):
                    with gr.Tabs():
                        with gr.TabItem("Revenue & Growth"):
                            revenue_growth_chart = gr.Plot(label="Revenue Growth")
                            key_metrics_income_chart = gr.Plot(label="Key Income Metrics")

                        with gr.TabItem("Profitability"):
                            margin_chart = gr.Plot(label="Margin Analysis")
                            profitability_chart = gr.Plot(label="Profitability Ratios")

                        with gr.TabItem("Balance Sheet Analysis"):
                            key_metrics_balance_chart = gr.Plot(label="Key Balance Sheet Metrics")
                            efficiency_chart = gr.Plot(label="Efficiency Ratios")

                        with gr.TabItem("Cash Flow Analysis"):
                            key_metrics_cash_chart = gr.Plot(label="Key Cash Flow Metrics")

    # Define function to clear figures when app is closed
    def on_close():
        plt.close('all')

    # Register the function to be called when the app is closed
    app.load(on_close)

    analyze_button.click(
        analyze_stock,
        inputs=[ticker_input, growth_rate, discount_rate, projection_years, format_type],
        outputs=[
            metrics_output, 
            valuation_output, 
            price_chart,
            annual_income_output,
            annual_balance_output,
            annual_cash_flow_output,
            quarterly_income_output,
            quarterly_balance_output,
            quarterly_cash_flow_output,
            income_chart,
            balance_chart,
            cash_flow_chart,
            q_income_chart,
            q_balance_chart,
            q_cash_flow_chart,
            spider_chart,
            revenue_growth_chart,
            margin_chart,
            key_metrics_income_chart,
            key_metrics_balance_chart,
            key_metrics_cash_chart,
            profitability_chart
        ]
    )

# Launch the app
if __name__ == "__main__":
    app.launch()
