import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import matplotlib.dates as mdates
from matplotlib.patches import Circle, RegularPolygon
from matplotlib.path import Path
from matplotlib.projections.polar import PolarAxes
from matplotlib.projections import register_projection
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D
import locale
import matplotlib.ticker as mtick

# Set locale for number formatting
try:
    locale.setlocale(locale.LC_ALL, '')
except:
    pass  # Fallback if locale setting fails

def format_number(number, precision=2, currency_symbol='$', format_type='auto'):
    """
    Format large numbers with K, M, B, T suffixes or with commas

    Parameters:
    - number: The number to format
    - precision: Decimal precision
    - currency_symbol: Currency symbol to use
    - format_type: 'auto', 'suffix', 'comma', 'millions', 'billions'
    """
    if number is None or number == 'N/A':
        return 'N/A'

    try:
        number = float(number)
    except:
        return str(number)

    # Handle negative numbers
    is_negative = number < 0
    abs_number = abs(number)

    # Format based on type
    if format_type == 'comma':
        # Format with commas
        try:
            formatted = locale.format_string(f"%.{precision}f", abs_number, grouping=True)
        except:
            # Fallback if locale formatting fails
            formatted = f"{abs_number:,.{precision}f}"
    elif format_type == 'millions':
        # Always format in millions
        formatted = f"{abs_number / 1_000_000:.{precision}f}M"
    elif format_type == 'billions':
        # Always format in billions
        formatted = f"{abs_number / 1_000_000_000:.{precision}f}B"
    else:  # 'auto' or 'suffix'
        # Format with appropriate suffix based on magnitude
        if abs_number >= 1_000_000_000_000:
            formatted = f"{abs_number / 1_000_000_000_000:.{precision}f}T"
        elif abs_number >= 1_000_000_000:
            formatted = f"{abs_number / 1_000_000_000:.{precision}f}B"
        elif abs_number >= 1_000_000:
            formatted = f"{abs_number / 1_000_000:.{precision}f}M"
        elif abs_number >= 1_000:
            formatted = f"{abs_number / 1_000:.{precision}f}K"
        else:
            formatted = f"{abs_number:.{precision}f}"

    # Add negative sign if needed
    if is_negative:
        return f"-{currency_symbol}{formatted}"
    else:
        return f"{currency_symbol}{formatted}"

def format_percentage(number, precision=2):
    """Format number as percentage"""
    if number is None or number == 'N/A':
        return 'N/A'

    try:
        number = float(number)
        return f"{number:.{precision}f}%"
    except:
        return str(number)

def create_price_chart(price_history):
    """Create a price chart from historical data"""
    # Close any existing figures to prevent memory issues
    plt.close('all')

    if price_history is None or len(price_history) == 0:
        fig = plt.figure(figsize=(10, 6))
        plt.text(0.5, 0.5, "No price history data available", 
                 horizontalalignment='center', verticalalignment='center', fontsize=14)
        plt.axis('off')
        return fig

    fig = plt.figure(figsize=(10, 6))

    # Calculate moving averages if enough data points
    if len(price_history) > 50:
        ma50 = price_history.rolling(window=50).mean()
        ma200 = price_history.rolling(window=min(200, len(price_history))).mean()

        plt.plot(price_history.index, price_history.values, label='Price')
        plt.plot(ma50.index, ma50.values, label='50-Day MA', linestyle='--')
        plt.plot(ma200.index, ma200.values, label='200-Day MA', linestyle='-.')
        plt.legend()
    else:
        plt.plot(price_history.index, price_history.values)

    # Format x-axis to show dates nicely
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.gcf().autofmt_xdate()

    # Add grid and labels
    plt.title('Stock Price History', fontsize=14)
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    return fig

def format_metrics_table(metrics):
    """Format metrics for display in a table"""
    formatted_metrics = {}
    currency_symbol = metrics.get('Currency Symbol', '$')

    for key, value in metrics.items():
        if key == 'Market Cap':
            formatted_metrics[key] = format_number(value, currency_symbol=currency_symbol)
        elif key in ['Dividend Yield (%)', 'Profit Margin', 'Operating Margin', 'ROE', 'ROA', 'Revenue Growth', 
                    'Payout Ratio', 'Earnings Growth', 'EBITDA Margins', 'Gross Margins'] or key.endswith('(%)'):
            formatted_metrics[key] = format_percentage(value)
        elif key in ['Current Price', 'EPS', '52 Week High', '52 Week Low', '50-Day MA', '200-Day MA', 
                    'Revenue Per Share', 'Target Mean Price', 'Free Cash Flow', 'Operating Cash Flow']:
            if value != 'N/A':
                formatted_metrics[key] = f"{currency_symbol}{value:.2f}"
            else:
                formatted_metrics[key] = value
        elif key in ['P/E Ratio', 'P/B Ratio', 'Forward P/E', 'PEG Ratio', 'Debt to Equity', 
                    'Current Ratio', 'Quick Ratio', 'Beta', 'EV/EBITDA', 'EV/Revenue']:
            if value != 'N/A':
                formatted_metrics[key] = f"{value:.2f}"
            else:
                formatted_metrics[key] = value
        else:
            formatted_metrics[key] = value

    return formatted_metrics

def format_financial_statement(statement, statement_type, currency_symbol='$', format_type='millions'):
    """
    Format financial statement for display

    Parameters:
    - statement: The financial statement DataFrame
    - statement_type: Type of statement (for title)
    - currency_symbol: Currency symbol to use
    - format_type: How to format numbers ('comma', 'millions', 'billions', 'auto')
    """
    if statement is None or statement.empty:
        return pd.DataFrame()

    # Make a copy to avoid modifying the original
    df = statement.copy()

    # Format column names (dates)
    df.columns = [col.strftime('%Y-%m-%d') if isinstance(col, datetime) else str(col) for col in df.columns]

    # Format the values based on format_type
    if format_type == 'comma':
        # Format with commas
        for col in df.columns:
            df[col] = df[col].apply(lambda x: format_number(x, currency_symbol=currency_symbol, format_type='comma') if pd.notnull(x) else 'N/A')
    elif format_type == 'millions':
        # Convert to millions and format
        df = df / 1_000_000
        for col in df.columns:
            df[col] = df[col].apply(lambda x: f"{currency_symbol}{x:.2f}M" if pd.notnull(x) else 'N/A')
    elif format_type == 'billions':
        # Convert to billions and format
        df = df / 1_000_000_000
        for col in df.columns:
            df[col] = df[col].apply(lambda x: f"{currency_symbol}{x:.2f}B" if pd.notnull(x) else 'N/A')
    else:  # 'auto'
        # Determine appropriate scale based on data magnitude
        max_abs_val = abs(df.max().max())
        if max_abs_val >= 1_000_000_000:
            df = df / 1_000_000_000
            suffix = 'B'
        else:
            df = df / 1_000_000
            suffix = 'M'

        for col in df.columns:
            df[col] = df[col].apply(lambda x: f"{currency_symbol}{x:.2f}{suffix}" if pd.notnull(x) else 'N/A')

    return df

def prepare_financial_table(statement, currency_symbol='$', format_type='millions'):
    """
    Prepare financial statement for display in a table format

    Returns a dictionary with formatted data and metadata
    """
    if statement is None or statement.empty:
        return {"error": "No data available"}

    # Format the statement
    formatted_df = format_financial_statement(statement, "", currency_symbol, format_type)

    # Prepare data for display
    result = {
        "data": formatted_df.reset_index().to_dict('records'),
        "columns": [{"name": "Metric", "id": "index"}] + [{"name": col, "id": col} for col in formatted_df.columns],
        "format_type": format_type,
        "currency_symbol": currency_symbol
    }

    return result

def create_financial_chart(statement, title, chart_type='bar'):
    """Create a chart from financial statement data"""
    # Close any existing figures to prevent memory issues
    plt.close('all')

    if statement is None or statement.empty:
        fig = plt.figure(figsize=(12, 6))
        plt.text(0.5, 0.5, "No data available", 
                 horizontalalignment='center', verticalalignment='center', fontsize=14)
        plt.axis('off')
        return fig

    # Select key metrics based on statement type
    if 'Total Revenue' in statement.index:  # Income Statement
        metrics = ['Total Revenue', 'Gross Profit', 'Operating Income', 'Net Income']
    elif 'Total Assets' in statement.index:  # Balance Sheet
        metrics = ['Total Assets', 'Total Liabilities Net Minority Interest', 'Total Equity Gross Minority Interest']
    elif 'Operating Cash Flow' in statement.index:  # Cash Flow
        metrics = ['Operating Cash Flow', 'Free Cash Flow', 'Capital Expenditures']
    else:
        # Default to first 4 rows if specific metrics not found
        metrics = statement.index[:4]

    # Filter for selected metrics that exist in the statement
    metrics = [m for m in metrics if m in statement.index]

    if not metrics:
        fig = plt.figure(figsize=(12, 6))
        plt.text(0.5, 0.5, "No relevant metrics found", 
                 horizontalalignment='center', verticalalignment='center', fontsize=14)
        plt.axis('off')
        return fig

    # Get data for the selected metrics
    data = statement.loc[metrics]

    # Convert to millions for better readability
    data = data / 1_000_000

    # Create the chart
    fig = plt.figure(figsize=(12, 6))

    if chart_type == 'bar':
        ax = data.T.plot(kind='bar', ax=plt.gca(), width=0.8)

        # Add value labels on top of bars
        for container in ax.containers:
            ax.bar_label(container, fmt='%.1fM', fontsize=8)
    else:  # line chart
        ax = data.T.plot(kind='line', marker='o', ax=plt.gca())

        # Add value labels at data points
        for line, metric in zip(ax.get_lines(), metrics):
            x_data, y_data = line.get_data()
            for x, y in zip(x_data, y_data):
                ax.annotate(f'{y:.1f}M', (x, y), textcoords="offset points", 
                            xytext=(0,5), ha='center', fontsize=8)

    plt.title(title, fontsize=14)
    plt.ylabel('Millions ($)')
    plt.grid(True, alpha=0.3)
    plt.legend(loc='best')
    plt.tight_layout()

    return fig

def create_key_metrics_chart(statement, title, metrics_list, currency_symbol='$'):
    """Create a chart for specific key metrics from financial statements"""
    # Close any existing figures to prevent memory issues
    plt.close('all')

    if statement is None or statement.empty:
        fig = plt.figure(figsize=(12, 6))
        plt.text(0.5, 0.5, "No data available", 
                 horizontalalignment='center', verticalalignment='center', fontsize=14)
        plt.axis('off')
        return fig

    # Filter for selected metrics that exist in the statement
    available_metrics = [m for m in metrics_list if m in statement.index]

    if not available_metrics:
        fig = plt.figure(figsize=(12, 6))
        plt.text(0.5, 0.5, "No relevant metrics found", 
                 horizontalalignment='center', verticalalignment='center', fontsize=14)
        plt.axis('off')
        return fig

    # Get data for the selected metrics
    data = statement.loc[available_metrics]

    # Convert to millions for better readability
    data = data / 1_000_000

    # Create the chart
    fig = plt.figure(figsize=(12, 6))

    # Create a bar chart
    ax = data.T.plot(kind='bar', ax=plt.gca(), width=0.8)

    # Add value labels on top of bars
    for container in ax.containers:
        ax.bar_label(container, fmt=f'%.1fM', fontsize=8)

    plt.title(title, fontsize=14)
    plt.ylabel(f'Millions ({currency_symbol})')
    plt.grid(True, alpha=0.3)
    plt.legend(loc='best')
    plt.tight_layout()

    return fig

def create_growth_chart(statement, metric_name, title):
    """Create a growth rate chart for a specific metric"""
    # Close any existing figures to prevent memory issues
    plt.close('all')

    if statement is None or statement.empty or metric_name not in statement.index:
        fig = plt.figure(figsize=(10, 6))
        plt.text(0.5, 0.5, f"No data available for {metric_name}", 
                 horizontalalignment='center', verticalalignment='center', fontsize=14)
        plt.axis('off')
        return fig

    # Get data for the selected metric
    data = statement.loc[metric_name]

    # Calculate year-over-year growth rates
    growth_rates = data.pct_change(-1) * 100  # Multiply by -1 to get YoY since columns are in reverse chronological order

    # Create the chart
    fig = plt.figure(figsize=(10, 6))

    # Plot the growth rates
    ax = plt.gca()
    bars = ax.bar(growth_rates.index, growth_rates.values, color='teal', alpha=0.7)

    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        if not np.isnan(height):
            ax.text(bar.get_x() + bar.get_width()/2., height + (1 if height >= 0 else -5),
                    f'{height:.1f}%', ha='center', va='bottom' if height >= 0 else 'top', fontsize=9)

    # Format y-axis as percentage
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())

    # Add a horizontal line at y=0
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)

    plt.title(title, fontsize=14)
    plt.ylabel('Year-over-Year Growth (%)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    return fig

def create_margin_chart(statement, title):
    """Create a chart showing margin trends"""
    # Close any existing figures to prevent memory issues
    plt.close('all')

    # Check if we have the necessary data
    required_metrics = ['Total Revenue', 'Gross Profit', 'Operating Income', 'Net Income']
    if statement is None or statement.empty or not all(metric in statement.index for metric in required_metrics):
        fig = plt.figure(figsize=(10, 6))
        plt.text(0.5, 0.5, "Insufficient data for margin analysis", 
                 horizontalalignment='center', verticalalignment='center', fontsize=14)
        plt.axis('off')
        return fig

    # Get data for the required metrics
    revenue = statement.loc['Total Revenue']
    gross_profit = statement.loc['Gross Profit']
    operating_income = statement.loc['Operating Income']
    net_income = statement.loc['Net Income']

    # Calculate margins
    gross_margin = (gross_profit / revenue) * 100
    operating_margin = (operating_income / revenue) * 100
    net_margin = (net_income / revenue) * 100

    # Create DataFrame for plotting
    margins_df = pd.DataFrame({
        'Gross Margin': gross_margin,
        'Operating Margin': operating_margin,
        'Net Margin': net_margin
    })

    # Create the chart
    fig = plt.figure(figsize=(10, 6))

    # Plot margins
    ax = margins_df.plot(kind='line', marker='o', ax=plt.gca())

    # Format y-axis as percentage
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())

    # Add value labels at data points
    for line, margin_type in zip(ax.get_lines(), margins_df.columns):
        x_data, y_data = line.get_data()
        for x, y in zip(x_data, y_data):
            ax.annotate(f'{y:.1f}%', (x, y), textcoords="offset points", 
                        xytext=(0,5), ha='center', fontsize=8)

    plt.title(title, fontsize=14)
    plt.ylabel('Margin (%)')
    plt.grid(True, alpha=0.3)
    plt.legend(loc='best')
    plt.tight_layout()

    return fig

def radar_factory(num_vars, frame='circle'):
    """Create a radar chart with `num_vars` axes."""
    # Calculate evenly-spaced axis angles
    theta = np.linspace(0, 2*np.pi, num_vars, endpoint=False)

    class RadarAxes(PolarAxes):
        name = 'radar'

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # Rotate plot so that first axis is at the top
            self.set_theta_zero_location('N')

        def fill(self, *args, closed=True, **kwargs):
            """Override fill so that line is closed by default"""
            return super().fill(closed=closed, *args, **kwargs)

        def plot(self, *args, **kwargs):
            """Override plot so that line is closed by default"""
            lines = super().plot(*args, **kwargs)
            for line in lines:
                self._close_line(line)
            return lines

        def _close_line(self, line):
            x, y = line.get_data()
            # FIXME: markers at x[0], y[0] get doubled-up
            if x[0] != x[-1]:
                x = np.append(x, x[0])
                y = np.append(y, y[0])
                line.set_data(x, y)

        def set_varlabels(self, labels):
            self.set_thetagrids(np.degrees(theta), labels)

        def _gen_axes_patch(self):
            # The Axes patch must be centered at (0.5, 0.5) and of radius 0.5
            # in axes coordinates.
            if frame == 'circle':
                return Circle((0.5, 0.5), 0.5)
            elif frame == 'polygon':
                return RegularPolygon((0.5, 0.5), num_vars, radius=0.5, orientation=np.pi/2)
            else:
                raise ValueError("Unknown value for 'frame': %s" % frame)

        def _gen_axes_spines(self):
            if frame == 'circle':
                return super()._gen_axes_spines()
            elif frame == 'polygon':
                # spine_type must be 'left'/'right'/'top'/'bottom'/'circle'.
                spine = Spine(axes=self,
                              spine_type='circle',
                              path=Path.unit_regular_polygon(num_vars))
                # unit_regular_polygon returns a polygon of radius 1 centered at
                # (0, 0) but we want a polygon of radius 0.5 centered at (0.5,
                # 0.5) in axes coordinates.
                spine.set_transform(Affine2D().scale(.5).translate(.5, .5)
                                    + self.transAxes)
                return {'polar': spine}
            else:
                raise ValueError("Unknown value for 'frame': %s" % frame)

    # Register the projection with Matplotlib
    register_projection(RadarAxes)
    return theta

def create_spider_chart(metrics, title="Financial Metrics Comparison"):
    """Create a spider/radar chart for key financial metrics"""
    # Close any existing figures to prevent memory issues
    plt.close('all')

    # Select metrics to display on the spider chart - expanded list
    spider_metrics = {
        'P/E Ratio': metrics.get('P/E Ratio', 'N/A'),
        'P/B Ratio': metrics.get('P/B Ratio', 'N/A'),
        'EV/EBITDA': metrics.get('EV/EBITDA', 'N/A'),
        'PEG Ratio': metrics.get('PEG Ratio', 'N/A'),
        'ROE (%)': metrics.get('ROE', 'N/A'),
        'ROA (%)': metrics.get('ROA', 'N/A'),
        'Profit Margin (%)': metrics.get('Profit Margin', 'N/A'),
        'Operating Margin (%)': metrics.get('Operating Margin', 'N/A'),
        'Debt to Equity': metrics.get('Debt to Equity', 'N/A'),
        'Current Ratio': metrics.get('Current Ratio', 'N/A'),
        'Dividend Yield (%)': metrics.get('Dividend Yield (%)', 'N/A'),
        'Revenue Growth (%)': metrics.get('Revenue Growth', 'N/A')
    }

    # Filter out N/A values and prepare data
    filtered_metrics = {k: v for k, v in spider_metrics.items() if v != 'N/A' and v is not None}

    if len(filtered_metrics) < 3:
        # Not enough metrics for a meaningful spider chart
        fig = plt.figure(figsize=(10, 10))
        plt.text(0.5, 0.5, "Insufficient data for spider chart", 
                 horizontalalignment='center', verticalalignment='center', fontsize=14)
        plt.axis('off')
        return fig

    # Prepare data for radar chart
    categories = list(filtered_metrics.keys())
    N = len(categories)

    # Create radar chart
    theta = radar_factory(N, frame='polygon')

    # Normalize values for better visualization
    values = list(filtered_metrics.values())

    # Define normalization parameters for each metric
    normalization_params = {
        'P/E Ratio': {'better': 'lower', 'max': 50, 'min': 0},
        'P/B Ratio': {'better': 'lower', 'max': 10, 'min': 0},
        'EV/EBITDA': {'better': 'lower', 'max': 20, 'min': 0},
        'PEG Ratio': {'better': 'lower', 'max': 3, 'min': 0},
        'ROE (%)': {'better': 'higher', 'max': 30, 'min': 0},
        'ROA (%)': {'better': 'higher', 'max': 15, 'min': 0},
        'Profit Margin (%)': {'better': 'higher', 'max': 30, 'min': 0},
        'Operating Margin (%)': {'better': 'higher', 'max': 30, 'min': 0},
        'Debt to Equity': {'better': 'lower', 'max': 3, 'min': 0},
        'Current Ratio': {'better': 'higher', 'max': 3, 'min': 0},
        'Dividend Yield (%)': {'better': 'higher', 'max': 10, 'min': 0},
        'Revenue Growth (%)': {'better': 'higher', 'max': 30, 'min': 0}
    }

    # Normalize values
    normalized = []
    for i, (cat, val) in enumerate(zip(categories, values)):
        params = normalization_params.get(cat, {'better': 'higher', 'max': 100, 'min': 0})

        # Clip value to min/max range
        val = max(min(val, params['max']), params['min'])

        # Normalize to 0-1 scale
        if params['better'] == 'lower':
            # For metrics where lower is better, invert the scale
            norm_val = 1 - ((val - params['min']) / (params['max'] - params['min']))
        else:
            # For metrics where higher is better
            norm_val = (val - params['min']) / (params['max'] - params['min'])

        normalized.append(norm_val)

    # Create the figure
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='radar'))

    # Plot the data
    ax.plot(theta, normalized, 'o-', linewidth=2)
    ax.fill(theta, normalized, alpha=0.25)

    # Set labels
    ax.set_varlabels(categories)

    # Add values to the plot
    for i, (angle, radius) in enumerate(zip(theta, normalized)):
        ax.text(angle, radius + 0.1, f"{values[i]:.1f}", 
                horizontalalignment='center', verticalalignment='center')

    # Add title
    plt.title(title, position=(0.5, 1.1), size=15)

    # Add a reference circle at 0.5
    ax.plot(theta, [0.5]*N, '--', color='gray', alpha=0.75, linewidth=1)

    return fig

def create_multi_year_growth_chart(statement, metrics, title, currency_symbol='$'):
    """Create a chart showing growth of multiple metrics over years"""
    # Close any existing figures to prevent memory issues
    plt.close('all')

    if statement is None or statement.empty:
        fig = plt.figure(figsize=(12, 6))
        plt.text(0.5, 0.5, "No data available", 
                 horizontalalignment='center', verticalalignment='center', fontsize=14)
        plt.axis('off')
        return fig

    # Filter for metrics that exist in the statement
    available_metrics = [m for m in metrics if m in statement.index]

    if not available_metrics:
        fig = plt.figure(figsize=(12, 6))
        plt.text(0.5, 0.5, "No relevant metrics found", 
                 horizontalalignment='center', verticalalignment='center', fontsize=14)
        plt.axis('off')
        return fig

    # Get data for the selected metrics
    data = statement.loc[available_metrics]

    # Convert to billions for better readability
    data = data / 1_000_000_000

    # Create the chart
    fig = plt.figure(figsize=(12, 6))

    # Plot as line chart
    ax = data.T.plot(kind='line', marker='o', ax=plt.gca())

    # Add value labels at data points
    for line, metric in zip(ax.get_lines(), available_metrics):
        x_data, y_data = line.get_data()
        for x, y in zip(x_data, y_data):
            ax.annotate(f'{y:.1f}B', (x, y), textcoords="offset points", 
                        xytext=(0,5), ha='center', fontsize=8)

    plt.title(title, fontsize=14)
    plt.ylabel(f'Billions ({currency_symbol})')
    plt.grid(True, alpha=0.3)
    plt.legend(loc='best')
    plt.tight_layout()

    return fig

def create_ratio_chart(statement, title, ratio_type='profitability'):
    """Create a chart showing financial ratios over time"""
    # Close any existing figures to prevent memory issues
    plt.close('all')

    if statement is None or statement.empty:
        fig = plt.figure(figsize=(10, 6))
        plt.text(0.5, 0.5, "No data available", 
                 horizontalalignment='center', verticalalignment='center', fontsize=14)
        plt.axis('off')
        return fig

    # Define metrics based on ratio type
    if ratio_type == 'profitability':
        if 'Total Revenue' in statement.index and 'Net Income' in statement.index:
            revenue = statement.loc['Total Revenue']
            net_income = statement.loc['Net Income']
            net_margin = (net_income / revenue) * 100

            if 'Gross Profit' in statement.index and 'Operating Income' in statement.index:
                gross_profit = statement.loc['Gross Profit']
                operating_income = statement.loc['Operating Income']

                gross_margin = (gross_profit / revenue) * 100
                operating_margin = (operating_income / revenue) * 100

                # Create DataFrame for plotting
                ratios_df = pd.DataFrame({
                    'Gross Margin': gross_margin,
                    'Operating Margin': operating_margin,
                    'Net Margin': net_margin
                })
            else:
                # Only net margin available
                ratios_df = pd.DataFrame({
                    'Net Margin': net_margin
                })
        else:
            fig = plt.figure(figsize=(10, 6))
            plt.text(0.5, 0.5, "Insufficient data for profitability ratios", 
                     horizontalalignment='center', verticalalignment='center', fontsize=14)
            plt.axis('off')
            return fig

    elif ratio_type == 'efficiency':
        if 'Total Assets' in statement.index and 'Net Income' in statement.index:
            assets = statement.loc['Total Assets']
            net_income = statement.loc['Net Income']
            roa = (net_income / assets) * 100

            if 'Total Equity Gross Minority Interest' in statement.index:
                equity = statement.loc['Total Equity Gross Minority Interest']
                roe = (net_income / equity) * 100

                # Create DataFrame for plotting
                ratios_df = pd.DataFrame({
                    'Return on Assets': roa,
                    'Return on Equity': roe
                })
            else:
                # Only ROA available
                ratios_df = pd.DataFrame({
                    'Return on Assets': roa
                })
        else:
            fig = plt.figure(figsize=(10, 6))
            plt.text(0.5, 0.5, "Insufficient data for efficiency ratios", 
                     horizontalalignment='center', verticalalignment='center', fontsize=14)
            plt.axis('off')
            return fig

    elif ratio_type == 'liquidity':
        if 'Current Assets' in statement.index and 'Current Liabilities' in statement.index:
            current_assets = statement.loc['Current Assets']
            current_liabilities = statement.loc['Current Liabilities']
            current_ratio = current_assets / current_liabilities

            if 'Inventory' in statement.index:
                inventory = statement.loc['Inventory']
                quick_ratio = (current_assets - inventory) / current_liabilities

                # Create DataFrame for plotting
                ratios_df = pd.DataFrame({
                    'Current Ratio': current_ratio,
                    'Quick Ratio': quick_ratio
                })
            else:
                # Only current ratio available
                ratios_df = pd.DataFrame({
                    'Current Ratio': current_ratio
                })
        else:
            fig = plt.figure(figsize=(10, 6))
            plt.text(0.5, 0.5, "Insufficient data for liquidity ratios", 
                     horizontalalignment='center', verticalalignment='center', fontsize=14)
            plt.axis('off')
            return fig

    else:  # Default case
        fig = plt.figure(figsize=(10, 6))
        plt.text(0.5, 0.5, f"Unknown ratio type: {ratio_type}", 
                 horizontalalignment='center', verticalalignment='center', fontsize=14)
        plt.axis('off')
        return fig

    # Create the chart
    fig = plt.figure(figsize=(10, 6))

    # Plot ratios
    ax = ratios_df.plot(kind='line', marker='o', ax=plt.gca())

    # Format y-axis as percentage for profitability and efficiency ratios
    if ratio_type in ['profitability', 'efficiency']:
        ax.yaxis.set_major_formatter(mtick.PercentFormatter())

    # Add value labels at data points
    for line, ratio_name in zip(ax.get_lines(), ratios_df.columns):
        x_data, y_data = line.get_data()
        for x, y in zip(x_data, y_data):
            if ratio_type in ['profitability', 'efficiency']:
                label = f'{y:.1f}%'
            else:
                label = f'{y:.2f}'
            ax.annotate(label, (x, y), textcoords="offset points", 
                        xytext=(0,5), ha='center', fontsize=8)

    plt.title(title, fontsize=14)

    if ratio_type == 'profitability':
        plt.ylabel('Margin (%)')
    elif ratio_type == 'efficiency':
        plt.ylabel('Return (%)')
    elif ratio_type == 'liquidity':
        plt.ylabel('Ratio')

    plt.grid(True, alpha=0.3)
    plt.legend(loc='best')
    plt.tight_layout()

    return fig
