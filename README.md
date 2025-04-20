# Stock DCF Valuation Tool

A comprehensive financial analysis application that provides DCF valuation and in-depth financial metrics visualization for any publicly traded company.

https://huggingface.co/spaces/vikramlingam/DCF-app

## Features

- **DCF Valuation**: Calculate intrinsic value using customizable growth rates (up to 250%), discount rates, and projection periods
- **Financial Statements**: View and analyze annual and quarterly income statements, balance sheets, and cash flow statements
- **Interactive Visualizations**: Explore key financial metrics through charts and graphs
- **Spider Chart Analysis**: Compare multiple financial ratios in a single visualization
- **Growth & Margin Analysis**: Track revenue growth, profit margins, and other key performance indicators
- **International Support**: Works with US stocks and international markets (including Indian NSE/BSE)
- **Flexible Number Formatting**: Display values in millions, billions, or with comma separators

## Installation

\`\`\`bash
# Clone the repository
git clone https://github.com/vikramlingam/DCF_App.git

cd DCF_App

# Install dependencies
pip install -r requirements.txt

# Run the application
python3 gradio_app.py
\`\`\`

## Usage

1. Enter a stock ticker symbol (e.g., AAPL, MSFT, TATAMOTORS.NS, ITC.NS etc.)
2. Adjust DCF parameters:
   - Growth Rate: Expected annual growth rate (0-250%)
   - Discount Rate: Required rate of return (5-20%)
   - Projection Years: Number of years to project (1-10)
3. Select number format (auto, comma, millions, billions)
4. Click "Analyze Stock" to generate the valuation and analysis

## Project Structure

- `gradio_app.py`: Main application with Gradio interface
- `data_fetcher.py`: Handles data retrieval from Yahoo Finance
- `valuation.py`: Contains DCF valuation logic
- `utils.py`: Visualization and formatting utilities
- `requirements.txt`: Required Python packages

## Technical Details

### DCF Valuation Methodology

The application uses a two-stage Discounted Cash Flow model:
1. **Growth Stage**: Projects Free Cash Flow (FCF) for the specified number of years using the user-defined growth rate
2. **Terminal Stage**: Calculates a terminal value using a sustainable long-term growth rate (capped at 4%)
3. **Present Value**: Discounts all future cash flows to present value using the specified discount rate

For companies with negative FCF, the model uses alternative approaches to provide a reasonable valuation estimate.

### Data Sources

All financial data is retrieved from Yahoo Finance using the `yfinance` library, providing:
- Historical price data
- Key financial metrics
- Annual and quarterly financial statements
- Market data (P/E, P/B, market cap, etc.)

### Visualization Capabilities

- **Price Charts**: Historical price with moving averages
- **Financial Statement Charts**: Bar charts of key line items
- **Spider Charts**: Radar visualization of 12 key financial metrics
- **Growth Charts**: Year-over-year growth visualization
- **Margin Analysis**: Gross, operating, and net margin trends
- **Ratio Charts**: Profitability and efficiency ratios

## Handling Edge Cases

The application includes robust error handling for:
- Missing quarterly data
- Negative cash flows
- Limited price history
- International stocks with different reporting standards

## Deployment

This application can be deployed as:
- A local web application
- A Flask web service
- A Hugging Face Space (see deployment instructions below)

### Deploying to Hugging Face Spaces

1. Create a Hugging Face account at [huggingface.co](https://huggingface.co/)
2. Create a new Space with Gradio SDK
3. Upload all project files
4. Rename main file to `app.py` if using web interface
5. Hugging Face will automatically build and deploy your application

## Requirements

- Python 3.8+
- gradio>=3.50.2
- yfinance>=0.2.28
- pandas>=2.0.0
- numpy>=1.24.0
- matplotlib>=3.7.0

## License

MIT License

## Acknowledgements

- [Yahoo Finance](https://finance.yahoo.com/) for financial data
- [Gradio](https://www.gradio.app/) for the web interface
- [Matplotlib](https://matplotlib.org/) for visualization capabilities

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
