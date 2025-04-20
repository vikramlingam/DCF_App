import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class DataFetcher:
    def __init__(self):
        # Cache to store fetched data
        self.cache = {}

    def get_ticker_info(self, ticker):
        """Get basic information about a ticker"""
        cache_key = f"{ticker}_info"
        if cache_key in self.cache:
            return self.cache[cache_key]

        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            self.cache[cache_key] = info
            return info
        except Exception as e:
            print(f"Error fetching info for {ticker}: {str(e)}")
            return {}

    def get_price_history(self, ticker, period="1y"):
        """Get historical price data for a ticker"""
        cache_key = f"{ticker}_price_{period}"
        if cache_key in self.cache:
            return self.cache[cache_key]

        try:
            stock = yf.Ticker(ticker)
            history = stock.history(period=period)

            if history.empty:
                # Try with a shorter period if 1y fails
                history = stock.history(period="6mo")

            if history.empty:
                # Try with an even shorter period if 6mo fails
                history = stock.history(period="3mo")

            if not history.empty:
                # Use Close prices
                price_series = history['Close']
                self.cache[cache_key] = price_series
                return price_series
            else:
                print(f"No price history available for {ticker}")
                return pd.Series()
        except Exception as e:
            print(f"Error fetching price history for {ticker}: {str(e)}")
            return pd.Series()

    def get_key_metrics(self, ticker):
        """Get key financial metrics for a ticker"""
        cache_key = f"{ticker}_metrics"
        if cache_key in self.cache:
            return self.cache[cache_key]

        try:
            stock = yf.Ticker(ticker)
            info = stock.info

            # Determine currency symbol based on country or exchange
            currency_symbol = '$'  # Default to USD
            if 'currency' in info:
                if info['currency'] == 'INR':
                    currency_symbol = '₹'
                elif info['currency'] == 'EUR':
                    currency_symbol = '€'
                elif info['currency'] == 'GBP':
                    currency_symbol = '£'
                elif info['currency'] == 'JPY':
                    currency_symbol = '¥'

            # Check if it's an Indian stock based on ticker suffix
            if ticker.endswith('.NS') or ticker.endswith('.BO'):
                currency_symbol = '₹'

            # Extract key metrics
            metrics = {
                'Company Name': info.get('longName', 'N/A'),
                'Sector': info.get('sector', 'N/A'),
                'Industry': info.get('industry', 'N/A'),
                'Country': info.get('country', 'N/A'),
                'Currency Symbol': currency_symbol,
                'Current Price': info.get('currentPrice', info.get('regularMarketPrice', 'N/A')),
                'Market Cap': info.get('marketCap', 'N/A'),
                'P/E Ratio': info.get('trailingPE', 'N/A'),
                'Forward P/E': info.get('forwardPE', 'N/A'),
                'P/B Ratio': info.get('priceToBook', 'N/A'),
                'EV/EBITDA': info.get('enterpriseToEbitda', 'N/A'),
                'EV/Revenue': info.get('enterpriseToRevenue', 'N/A'),
                'PEG Ratio': info.get('pegRatio', 'N/A'),
                'Dividend Yield (%)': info.get('dividendYield', 'N/A') * 100 if info.get('dividendYield') is not None else 'N/A',
                'EPS': info.get('trailingEps', 'N/A'),
                'Profit Margin': info.get('profitMargins', 'N/A') * 100 if info.get('profitMargins') is not None else 'N/A',
                'Operating Margin': info.get('operatingMargins', 'N/A') * 100 if info.get('operatingMargins') is not None else 'N/A',
                'ROE': info.get('returnOnEquity', 'N/A') * 100 if info.get('returnOnEquity') is not None else 'N/A',
                'ROA': info.get('returnOnAssets', 'N/A') * 100 if info.get('returnOnAssets') is not None else 'N/A',
                'Revenue Growth': info.get('revenueGrowth', 'N/A') * 100 if info.get('revenueGrowth') is not None else 'N/A',
                'Earnings Growth': info.get('earningsGrowth', 'N/A') * 100 if info.get('earningsGrowth') is not None else 'N/A',
                'Debt to Equity': info.get('debtToEquity', 'N/A') / 100 if info.get('debtToEquity') is not None else 'N/A',
                'Current Ratio': info.get('currentRatio', 'N/A'),
                'Quick Ratio': info.get('quickRatio', 'N/A'),
                'Beta': info.get('beta', 'N/A'),
                '52 Week High': info.get('fiftyTwoWeekHigh', 'N/A'),
                '52 Week Low': info.get('fiftyTwoWeekLow', 'N/A'),
                '50-Day MA': info.get('fiftyDayAverage', 'N/A'),
                '200-Day MA': info.get('twoHundredDayAverage', 'N/A'),
                'Shares Outstanding': info.get('sharesOutstanding', 'N/A'),
                'Free Cash Flow': info.get('freeCashflow', 'N/A'),
                'Operating Cash Flow': info.get('operatingCashflow', 'N/A'),
                'Revenue Per Share': info.get('revenuePerShare', 'N/A'),
                'Target Mean Price': info.get('targetMeanPrice', 'N/A'),
                'Payout Ratio': info.get('payoutRatio', 'N/A') * 100 if info.get('payoutRatio') is not None else 'N/A',
                'EBITDA Margins': info.get('ebitdaMargins', 'N/A') * 100 if info.get('ebitdaMargins') is not None else 'N/A',
                'Gross Margins': info.get('grossMargins', 'N/A') * 100 if info.get('grossMargins') is not None else 'N/A'
            }

            self.cache[cache_key] = metrics
            return metrics
        except Exception as e:
            print(f"Error fetching metrics for {ticker}: {str(e)}")
            return {'error': str(e), 'Currency Symbol': '$'}

    def get_financial_statements(self, ticker):
        """Get financial statements for a ticker"""
        cache_key = f"{ticker}_financials"
        if cache_key in self.cache:
            return self.cache[cache_key]

        try:
            stock = yf.Ticker(ticker)

            # Get annual financial statements
            income_stmt = stock.income_stmt
            balance_sheet = stock.balance_sheet
            cash_flow = stock.cashflow

            # Get quarterly financial statements with error handling
            try:
                quarterly_income_stmt = stock.quarterly_income_stmt
            except Exception as e:
                print(f"Error fetching quarterly income statement for {ticker}: {str(e)}")
                quarterly_income_stmt = pd.DataFrame()

            try:
                quarterly_balance_sheet = stock.quarterly_balance_sheet
            except Exception as e:
                print(f"Error fetching quarterly balance sheet for {ticker}: {str(e)}")
                quarterly_balance_sheet = pd.DataFrame()

            try:
                quarterly_cash_flow = stock.quarterly_cashflow
            except Exception as e:
                print(f"Error fetching quarterly cash flow for {ticker}: {str(e)}")
                quarterly_cash_flow = pd.DataFrame()

            # Package all statements
            financial_statements = {
                'income_stmt': income_stmt,
                'balance_sheet': balance_sheet,
                'cash_flow': cash_flow,
                'quarterly_income_stmt': quarterly_income_stmt,
                'quarterly_balance_sheet': quarterly_balance_sheet,
                'quarterly_cash_flow': quarterly_cash_flow
            }

            self.cache[cache_key] = financial_statements
            return financial_statements
        except Exception as e:
            print(f"Error fetching financial statements for {ticker}: {str(e)}")
            # Return empty DataFrames for all statements
            empty_df = pd.DataFrame()
            return {
                'income_stmt': empty_df,
                'balance_sheet': empty_df,
                'cash_flow': empty_df,
                'quarterly_income_stmt': empty_df,
                'quarterly_balance_sheet': empty_df,
                'quarterly_cash_flow': empty_df
            }

    def get_free_cash_flow(self, ticker):
        """Get the most recent free cash flow value"""
        try:
            # First try to get FCF directly from info
            metrics = self.get_key_metrics(ticker)
            if metrics.get('Free Cash Flow', 'N/A') != 'N/A':
                fcf = metrics.get('Free Cash Flow')
                # Handle negative FCF by using 0 as the base
                if fcf is not None and fcf < 0:
                    print(f"Warning: Negative FCF ({fcf}) for {ticker}, using 0 as base for DCF")
                    return 0
                return fcf

            # If not available, calculate from cash flow statement
            financial_statements = self.get_financial_statements(ticker)
            cash_flow = financial_statements['cash_flow']

            if cash_flow.empty:
                return None

            # Check if 'Free Cash Flow' is directly available
            if 'Free Cash Flow' in cash_flow.index:
                fcf = cash_flow.loc['Free Cash Flow', cash_flow.columns[0]]
                # Handle negative FCF by using 0 as the base
                if fcf is not None and fcf < 0:
                    print(f"Warning: Negative FCF ({fcf}) for {ticker}, using 0 as base for DCF")
                    return 0
                return fcf

            # If not, try to calculate it from components
            if 'Operating Cash Flow' in cash_flow.index and 'Capital Expenditure' in cash_flow.index:
                operating_cf = cash_flow.loc['Operating Cash Flow', cash_flow.columns[0]]
                capex = cash_flow.loc['Capital Expenditure', cash_flow.columns[0]]

                if pd.notnull(operating_cf) and pd.notnull(capex):
                    fcf = operating_cf + capex  # Note: capex is usually negative
                    # Handle negative FCF by using 0 as the base
                    if fcf is not None and fcf < 0:
                        print(f"Warning: Negative FCF ({fcf}) for {ticker}, using 0 as base for DCF")
                        return 0
                    return fcf

            return None
        except Exception as e:
            print(f"Error calculating free cash flow for {ticker}: {str(e)}")
            return None
