import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import minimize
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import openpyxl # Keep this import for clarity, though Pandas uses it under the hood

# --- Configuration & Constants ---
st.set_page_config(layout="wide")
st.title("📈 Efficient Frontier & Portfolio Optimization App")

EXCEL_FILE_PATH = "all_stocks_close_prices_2019_2023.xlsx"
RISK_FREE_RATE = 0 # Annual risk-free rate (e.g., 2%)
NUM_PORTFOLIOS_FRONTIER = 100 # Number of points to calculate for the frontier line

# --- Data Loading ---
@st.cache_data
def load_all_stock_data_from_excel(file_path):
    """Loads all stock data from the Excel file and parses dates."""
    try:
        data = pd.read_excel(file_path, parse_dates=['Date'], index_col='Date')
        for col in data.columns:
            data[col] = pd.to_numeric(data[col], errors='coerce')
        data = data.dropna(axis=1, how='all')
        data = data.dropna(axis=0, how='all')
        if data.empty:
            st.error(f"No data found in '{file_path}' after cleaning. Please check the file content.")
            return pd.DataFrame()
        st.success(f"Successfully loaded data from '{file_path}'.")
        return data
    except FileNotFoundError:
        st.error(f"Error: The file '{file_path}' was not found. "
                 f"Please make sure it's in the same directory as the app and named correctly.")
        return pd.DataFrame()
    except ValueError as ve:
        st.error(f"Error parsing dates in '{file_path}'. Ensure 'Date' column is in a recognizable format: {ve}")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error reading or processing Excel file '{file_path}': {e}")
        return pd.DataFrame()

ALL_STOCK_DATA = load_all_stock_data_from_excel(EXCEL_FILE_PATH)

# Determine available stocks and FULL date range from Excel
available_excel_stocks = []
min_data_date_from_excel, max_data_date_from_excel = None, None

if not ALL_STOCK_DATA.empty:
    available_excel_stocks = sorted(list(ALL_STOCK_DATA.columns))
    min_data_date_from_excel = ALL_STOCK_DATA.index.min().date() # This will be our fixed start date
    max_data_date_from_excel = ALL_STOCK_DATA.index.max().date() # This will be our fixed end date

    # --- REMOVED: default_user_start_date and default_user_end_date logic ---
    # --- REMOVED: actual_default_start_date and actual_default_end_date logic ---
    # We will directly use min_data_date_from_excel and max_data_date_from_excel
else:
    # Fallback if file loading fails, app will be limited but we still need to define these
    # so the app doesn't crash before the error messages about loading data are shown.
    min_data_date_from_excel = datetime.today().date() - timedelta(days=1)
    max_data_date_from_excel = datetime.today().date()
    st.error("Critical: Stock data could not be loaded. Analysis will not be possible.")


# --- Helper Functions for Portfolio Optimization ---
# ... (get_stock_data, calculate_returns_covariance, portfolio_performance, etc. remain UNCHANGED) ...
# Your existing helper functions are fine.

def get_stock_data(tickers, start_date, end_date):
    """Fetches historical prices for a list of tickers from the pre-loaded DataFrame."""
    if ALL_STOCK_DATA.empty:
        st.warning("No stock data loaded from Excel. Cannot proceed.")
        return pd.DataFrame()
    if start_date is None or end_date is None: # Should not happen with fixed dates, but good check
        st.error("Date range not determined. Cannot fetch stock data.")
        return pd.DataFrame()

    start_dt = datetime.combine(start_date, datetime.min.time())
    end_dt = datetime.combine(end_date, datetime.max.time())

    valid_tickers_in_request = [ticker for ticker in tickers if ticker in ALL_STOCK_DATA.columns]
    missing_tickers = [ticker for ticker in tickers if ticker not in ALL_STOCK_DATA.columns]

    if missing_tickers:
        st.warning(f"The following tickers were not found in the data file and will be ignored: {', '.join(missing_tickers)}")
    
    if not valid_tickers_in_request:
        st.error("None of the selected/profile tickers are available in the data file.")
        return pd.DataFrame()
    try:
        data_filtered_by_date = ALL_STOCK_DATA.loc[start_dt:end_dt]
        data_final_selection = data_filtered_by_date[valid_tickers_in_request]
    except KeyError as e:
        st.error(f"Error selecting data for tickers {valid_tickers_in_request} within the date range: {e}. ")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"An unexpected error occurred during data slicing: {e}")
        return pd.DataFrame()
    
    if isinstance(data_final_selection, pd.Series) and len(valid_tickers_in_request) == 1:
        data_final_selection = data_final_selection.to_frame(name=valid_tickers_in_request[0])
    
    return data_final_selection.dropna(how='any')

# ... (calculate_returns_covariance, portfolio_performance, neg_sharpe_ratio, etc. - NO CHANGES NEEDED HERE) ...
# ... (get_efficient_frontier_points, optimize_portfolio, display_portfolio_details, plot_efficient_frontier_plotly - NO CHANGES NEEDED HERE) ...
# ... (PREDEFINED_SETS - NO CHANGES NEEDED HERE) ...
# ... (global_x_min, global_x_max, update_global_ranges - NO CHANGES NEEDED HERE) ...
# ... (run_analysis function - NO CHANGES NEEDED INSIDE THE FUNCTION ITSELF, only how it's called) ...

# --- Existing functions (no changes needed inside them) ---
def calculate_returns_covariance(data):
    daily_returns = data.pct_change().dropna()
    if isinstance(daily_returns, pd.Series):
        daily_returns = daily_returns.to_frame(name=data.columns[0])
    if daily_returns.shape[0] < 2:
        st.warning("Not enough data points after calculating returns to perform analysis. Full data range might be too short or data is sparse.")
        return pd.DataFrame(), pd.Series(dtype=float), pd.DataFrame()
    mean_returns = daily_returns.mean() * 252
    cov_matrix = daily_returns.cov() * 252
    return daily_returns, mean_returns, cov_matrix

def portfolio_performance(weights, mean_returns, cov_matrix):
    portfolio_return = np.sum(mean_returns * weights)
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    sharpe_ratio = (portfolio_return - RISK_FREE_RATE) / portfolio_volatility if portfolio_volatility != 0 else -np.inf
    return portfolio_return, portfolio_volatility, sharpe_ratio

def neg_sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate):
    return -portfolio_performance(weights, mean_returns, cov_matrix)[2]

def portfolio_volatility_objective(weights, cov_matrix):
    return portfolio_performance(weights, np.zeros(len(weights)), cov_matrix)[1]

def get_efficient_frontier_points(mean_returns, cov_matrix, num_assets):
    results_list = []
    min_ret_individual = mean_returns.min()
    max_ret_individual = mean_returns.max()
    if max_ret_individual <= min_ret_individual:
        return pd.DataFrame(columns=['Return', 'Volatility', 'Sharpe', 'Weights'])
    if max_ret_individual > 0 and min_ret_individual < 0:
        target_returns = np.linspace(min_ret_individual, max_ret_individual, NUM_PORTFOLIOS_FRONTIER)
    elif max_ret_individual <=0:
        target_returns = np.linspace(min_ret_individual * 1.2, min_ret_individual * 0.8 , NUM_PORTFOLIOS_FRONTIER)
    else:
        target_returns = np.linspace(min_ret_individual * 0.8, max_ret_individual * 1.2, NUM_PORTFOLIOS_FRONTIER)
    initial_guess = np.array([1./num_assets] * num_assets)
    bounds = tuple((0, 1) for _ in range(num_assets))
    for target_return in target_returns:
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
            {'type': 'eq', 'fun': lambda w: portfolio_performance(w, mean_returns, cov_matrix)[0] - target_return}
        ]
        try:
            result = minimize(portfolio_volatility_objective, initial_guess, args=(cov_matrix,),
                              method='SLSQP', bounds=bounds, constraints=constraints, tol=1e-7)
            if result.success:
                weights = result.x
                p_ret, p_vol, p_sharpe = portfolio_performance(weights, mean_returns, cov_matrix)
                if abs(p_ret - target_return) < 0.015 and all(w >= -1e-5 for w in weights) and all(w <= 1+1e-5 for w in weights):
                    results_list.append({'Return': p_ret, 'Volatility': p_vol, 'Sharpe': p_sharpe, 'Weights': weights})
        except (ValueError, Exception) as e:
            pass
    if not results_list: return pd.DataFrame(columns=['Return', 'Volatility', 'Sharpe', 'Weights'])
    results_df = pd.DataFrame(results_list).sort_values(by=['Volatility', 'Return'], ascending=[True, False])
    efficient_portfolios = []
    min_vol_for_ret = np.inf
    results_df_sorted_return = results_df.sort_values(by=['Return', 'Volatility'], ascending=[False, True]) # Temp name change
    for i in range(len(results_df_sorted_return)):
        row = results_df_sorted_return.iloc[i]
        if row['Volatility'] <= min_vol_for_ret:
            efficient_portfolios.append(row)
            min_vol_for_ret = row['Volatility']
    if not efficient_portfolios: efficient_portfolios = results_df
    return pd.DataFrame(efficient_portfolios).sort_values(by='Volatility')

def optimize_portfolio(mean_returns, cov_matrix, num_assets, objective_type="min_variance"):
    initial_guess = np.array([1./num_assets] * num_assets)
    bounds = tuple((0, 1) for _ in range(num_assets))
    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
    target_fn, args_tuple = None, ()
    if objective_type == "min_variance":
        target_fn, args_tuple = portfolio_volatility_objective, (cov_matrix,)
    elif objective_type == "max_sharpe":
        target_fn, args_tuple = neg_sharpe_ratio, (mean_returns, cov_matrix, RISK_FREE_RATE)
    elif objective_type == "max_return":
        best_stock_idx = np.argmax(mean_returns.values)
        weights = np.zeros(num_assets); weights[best_stock_idx] = 1.0
        p_ret, p_vol, p_sharpe = portfolio_performance(weights, mean_returns, cov_matrix)
        return {'Return': p_ret, 'Volatility': p_vol, 'Sharpe': p_sharpe, 'Weights': weights}
    else: raise ValueError("Invalid objective_type")
    result = minimize(target_fn, initial_guess, args=args_tuple, method='SLSQP', bounds=bounds, constraints=constraints, tol=1e-7)
    if result.success:
        weights = result.x; weights[weights < 1e-5] = 0; weights = weights / np.sum(weights)
        p_ret, p_vol, p_sharpe = portfolio_performance(weights, mean_returns, cov_matrix)
        return {'Return': p_ret, 'Volatility': p_vol, 'Sharpe': p_sharpe, 'Weights': weights}
    return None

def display_portfolio_details(portfolio_dict, tickers, title):
    st.subheader(title)
    if portfolio_dict and 'Weights' in portfolio_dict and portfolio_dict['Weights'] is not None:
        st.write(f"Expected Annual Return: {portfolio_dict['Return']:.2%}")
        st.write(f"Annual Volatility (Risk): {portfolio_dict['Volatility']:.2%}")
        st.write(f"Sharpe Ratio: {portfolio_dict['Sharpe']:.2f}")
        st.write("Optimal Weights:")
        weights_df = pd.DataFrame(portfolio_dict['Weights'], index=tickers, columns=['Weight'])
        st.dataframe(weights_df[weights_df['Weight'] > 1e-4].style.format("{:.2%}"))
    else: st.write("Could not calculate portfolio details.")

def plot_efficient_frontier_plotly(frontier_points_df, min_var_pt, max_sharpe_pt, max_ret_pt, individual_stocks_df, title, x_range=None, y_range=None):
    fig = go.Figure()
    if not frontier_points_df.empty:
        frontier_points_df = frontier_points_df.sort_values(by='Volatility')
        fig.add_trace(go.Scatter(x=frontier_points_df['Volatility'], y=frontier_points_df['Return'], mode='lines', name='Efficient Frontier', line=dict(color='blue', width=2)))
    if min_var_pt: fig.add_trace(go.Scatter(x=[min_var_pt['Volatility']], y=[min_var_pt['Return']], mode='markers', name='Min Variance', marker=dict(color='green', size=12, symbol='diamond')))
    if max_sharpe_pt: fig.add_trace(go.Scatter(x=[max_sharpe_pt['Volatility']], y=[max_sharpe_pt['Return']], mode='markers', name='Max Sharpe Ratio', marker=dict(color='red', size=12, symbol='star')))
    if max_ret_pt: fig.add_trace(go.Scatter(x=[max_ret_pt['Volatility']], y=[max_ret_pt['Return']], mode='markers', name='Max Return (on Frontier)', marker=dict(color='purple', size=12, symbol='cross')))
    if not individual_stocks_df.empty: fig.add_trace(go.Scatter(x=individual_stocks_df['Volatility'], y=individual_stocks_df['Return'], mode='markers', name='Individual Stocks', text=individual_stocks_df.index, marker=dict(color='grey', size=8, symbol='circle')))
    fig.update_layout(title=title, xaxis_title='Annual Volatility (Risk)', yaxis_title='Annual Expected Return', yaxis_tickformat=".2%", xaxis_tickformat=".2%", legend_title="Portfolio Points", height=500)
    if x_range and x_range[0] is not None and x_range[1] is not None: fig.update_xaxes(range=[x_range[0] * 0.95, x_range[1] * 1.05])
    if y_range and y_range[0] is not None and y_range[1] is not None:
        padding_y_min = y_range[0] * 0.95 if y_range[0] > 0 else y_range[0] * 1.05
        padding_y_max = y_range[1] * 1.05 if y_range[1] > 0 else y_range[1] * 0.95
        if padding_y_min == padding_y_max: padding_y_min -= 0.01; padding_y_max += 0.01
        fig.update_yaxes(range=[padding_y_min, padding_y_max])
    return fig

PREDEFINED_SETS = {
    "Risk-Averse Investor": ["JNJ", "GILD", "AMGN", "DUK", "WMT"],
    "Risk-Adjusted Investor": ["AAPL", "MSFT", "NVDA", "AMGN", "WMT"],
    "Risk-Seeker Investor": ["MSFT", "NVDA", "SO", "AAPL", "WMT"]
}



def run_analysis(tickers_to_analyze, start_date_obj, end_date_obj, plot_title_prefix, key_suffix):
    global global_x_min, global_x_max, global_y_min, global_y_max
    if not tickers_to_analyze:
        st.warning("Please select at least one stock or ensure profile has stocks.")
        return None, None, None, pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    data = get_stock_data(tickers_to_analyze, start_date_obj, end_date_obj)
    if data.empty or data.shape[0] < 2:
        st.error(f"Not enough data for selected stocks ({', '.join(tickers_to_analyze)}) for the full available date range. Check Excel file content.")
        return None, None, None, pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    actual_tickers_used = list(data.columns)
    if not actual_tickers_used:
        st.error("No valid tickers found in the data for analysis.")
        return None, None, None, pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    num_assets = len(actual_tickers_used)
    daily_returns, mean_returns, cov_matrix = calculate_returns_covariance(data)
    if daily_returns.empty or mean_returns.empty or cov_matrix.empty:
        st.error("Could not calculate returns or covariance. Check data quality or quantity for the full date range.")
        return None, None, None, pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    individual_stocks_list = []
    for i, ticker in enumerate(actual_tickers_used):
        stock_return = mean_returns.get(ticker)
        stock_volatility = np.sqrt(cov_matrix.iloc[i, i]) if cov_matrix.shape[0] > i and cov_matrix.shape[1] > i else 0
        if stock_return is not None:
            individual_stocks_list.append({'Ticker': ticker, 'Return': stock_return, 'Volatility': stock_volatility})
    individual_stocks_df = pd.DataFrame(individual_stocks_list).set_index('Ticker')
    if num_assets == 1:
        single_stock_ret = mean_returns.iloc[0]
        single_stock_vol = np.sqrt(cov_matrix.iloc[0,0])
        single_stock_sharpe = (single_stock_ret - RISK_FREE_RATE) / single_stock_vol if single_stock_vol != 0 else -np.inf
        pt_dict = {'Return': single_stock_ret, 'Volatility': single_stock_vol, 'Sharpe': single_stock_sharpe, 'Weights': np.array([1.0])}
        corr_matrix = daily_returns.corr() if not daily_returns.empty else pd.DataFrame(index=actual_tickers_used, columns=actual_tickers_used)
        return pt_dict, pt_dict, pt_dict, pd.DataFrame(), individual_stocks_df, corr_matrix
    min_var_portfolio = optimize_portfolio(mean_returns, cov_matrix, num_assets, "min_variance")
    max_sharpe_portfolio = optimize_portfolio(mean_returns, cov_matrix, num_assets, "max_sharpe")
    frontier_points_df = get_efficient_frontier_points(mean_returns, cov_matrix, num_assets)
    max_ret_portfolio = None
    if not frontier_points_df.empty:
        max_ret_portfolio = frontier_points_df.loc[frontier_points_df['Return'].idxmax()].to_dict()
    else:
        max_ret_portfolio = optimize_portfolio(mean_returns, cov_matrix, num_assets, "max_return")
    all_volatilities, all_returns = [], []
    for pt_data in [frontier_points_df, pd.DataFrame([min_var_portfolio]), pd.DataFrame([max_sharpe_portfolio]), pd.DataFrame([max_ret_portfolio]), individual_stocks_df.reset_index()]:
        if pt_data is not None and not pt_data.empty and 'Volatility' in pt_data.columns and 'Return' in pt_data.columns:
            all_volatilities.extend(pt_data['Volatility'].dropna().tolist())
            all_returns.extend(pt_data['Return'].dropna().tolist())
    

    corr_matrix = daily_returns.corr()
    return min_var_portfolio, max_sharpe_portfolio, max_ret_portfolio, frontier_points_df, individual_stocks_df, corr_matrix
# --- UI Layout & Logic ---

col1, col2 = st.columns(2)

# State management for results
if 'p1_results' not in st.session_state: st.session_state.p1_results = None
if 'p2_results' not in st.session_state: st.session_state.p2_results = None


with col1:
    st.header("Part 1: Custom Portfolio")
    if not available_excel_stocks:
        st.warning("No stocks available from Excel file for selection. Cannot select stocks for Part 1.")
        selected_tickers_p1 = []
    else:
        selected_tickers_p1 = st.multiselect(
            "Select up to 5 stocks:", options=available_excel_stocks,
            max_selections=5, key="p1_stocks"
        )
    
    # --- REMOVED: Date input for Part 1 ---

    # Ensure min_data_date_from_excel and max_data_date_from_excel are valid before using them
    analysis_possible_p1 = not ALL_STOCK_DATA.empty and min_data_date_from_excel and max_data_date_from_excel and min_data_date_from_excel <= max_data_date_from_excel

    if st.button("Analyze Custom Portfolio", key="p1_analyze_button", disabled=not analysis_possible_p1):
        if selected_tickers_p1: # Check if stocks are selected
            with st.spinner("Analyzing custom portfolio for full available date range..."):
                st.session_state.p1_results = run_analysis(
                    selected_tickers_p1,
                    min_data_date_from_excel, # Use full range
                    max_data_date_from_excel, # Use full range
                    "Custom Portfolio",
                    "p1"
                )
        elif not analysis_possible_p1:
             st.error("Cannot analyze: Data not loaded or date range from Excel is invalid.")
        else: # Stocks not selected
            st.warning("Please select stocks for Part 1.")
    
    plot_placeholder_p1 = st.empty()
    if st.session_state.p1_results:
        min_var_p1, max_sharpe_p1, max_ret_p1, frontier_p1, ind_stocks_p1, corr_p1 = st.session_state.p1_results
        actual_tickers_p1 = list(ind_stocks_p1.index) if ind_stocks_p1 is not None and not ind_stocks_p1.empty else selected_tickers_p1

        st.markdown("---")
        st.subheader("Portfolio Details (Custom Selection)")
        display_portfolio_details(min_var_p1, actual_tickers_p1, "Minimum Variance Portfolio")
        display_portfolio_details(max_sharpe_p1, actual_tickers_p1, "Maximum Sharpe Ratio Portfolio")
        display_portfolio_details(max_ret_p1, actual_tickers_p1, "Maximum Return Portfolio (on Frontier)")
        
        st.markdown("---")
        st.subheader("Correlation Matrix (Custom Selection)")
        if corr_p1 is not None and not corr_p1.empty:
            fig_corr_p1 = px.imshow(corr_p1, text_auto=".2f", aspect="auto", color_continuous_scale='RdBu_r', range_color=[-1,1])
            st.plotly_chart(fig_corr_p1, use_container_width=True)
        else:
            st.write("Not enough data for correlation matrix.")

with col2:
    st.header("Part 2: Investor Profile Portfolio")
    profile_options = list(PREDEFINED_SETS.keys())
    selected_profile = st.selectbox("Choose an investor profile:", options=profile_options, key="p2_profile")
    
    # --- REMOVED: Date input for Part 2 ---
    analysis_possible_p2 = not ALL_STOCK_DATA.empty and min_data_date_from_excel and max_data_date_from_excel and min_data_date_from_excel <= max_data_date_from_excel

    if st.button(f"Analyze {selected_profile} Portfolio", key="p2_analyze_button", disabled=not analysis_possible_p2):
        profile_tickers = PREDEFINED_SETS[selected_profile]
        with st.spinner(f"Analyzing {selected_profile} portfolio for full available date range..."):
            st.session_state.p2_results = run_analysis(
                profile_tickers,
                min_data_date_from_excel, # Use full range
                max_data_date_from_excel, # Use full range
                f"{selected_profile}",
                "p2"
            )
        # No need for else if profile_tickers is always valid from PREDEFINED_SETS
        # and analysis_possible_p2 handles data loading issues.

    plot_placeholder_p2 = st.empty()
    if st.session_state.p2_results:
        min_var_p2, max_sharpe_p2, max_ret_p2, frontier_p2, ind_stocks_p2, corr_p2 = st.session_state.p2_results
        actual_tickers_p2 = list(ind_stocks_p2.index) if ind_stocks_p2 is not None and not ind_stocks_p2.empty else PREDEFINED_SETS[selected_profile]

        st.markdown("---")
        st.subheader(f"Portfolio Details ({selected_profile})")
        display_portfolio_details(min_var_p2, actual_tickers_p2, "Minimum Variance Portfolio")
        display_portfolio_details(max_sharpe_p2, actual_tickers_p2, "Maximum Sharpe Ratio Portfolio")
        display_portfolio_details(max_ret_p2, actual_tickers_p2, "Maximum Return Portfolio (on Frontier)")

        st.markdown("---")
        st.subheader(f"Correlation Matrix ({selected_profile})")
        if corr_p2 is not None and not corr_p2.empty:
            fig_corr_p2 = px.imshow(corr_p2, text_auto=".2f", aspect="auto", color_continuous_scale='RdBu_r', range_color=[-1,1])
            st.plotly_chart(fig_corr_p2, use_container_width=True)
        else:
            st.write("Not enough data for correlation matrix.")


# --- Plotting with synchronized axes ---
if st.session_state.p1_results:
    min_var_p1, max_sharpe_p1, max_ret_p1, frontier_p1, ind_stocks_p1, _ = st.session_state.p1_results
    title_p1 = "Efficient Frontier: Custom Portfolio (Full Data Range)"
    if ind_stocks_p1 is not None and len(ind_stocks_p1) == 1: title_p1 = "Single Stock Performance: Custom Portfolio (Full Data Range)"
    _min_var_p1 = min_var_p1 if min_var_p1 and 'Volatility' in min_var_p1 else None
    _max_sharpe_p1 = max_sharpe_p1 if max_sharpe_p1 and 'Volatility' in max_sharpe_p1 else None
    _max_ret_p1 = max_ret_p1 if max_ret_p1 and 'Volatility' in max_ret_p1 else None
    fig_p1 = plot_efficient_frontier_plotly(
        frontier_p1, _min_var_p1, _max_sharpe_p1, _max_ret_p1, ind_stocks_p1, title_p1,
    )
    plot_placeholder_p1.plotly_chart(fig_p1, use_container_width=True)

if st.session_state.p2_results:
    min_var_p2, max_sharpe_p2, max_ret_p2, frontier_p2, ind_stocks_p2, _ = st.session_state.p2_results
    # Ensure selected_profile is defined when this block is reached
    # It should be from the selectbox state. If there's a chance it's not (e.g. first run with no p2 analysis yet)
    # you might need a default or check. However, this block only runs if p2_results exists.
    profile_name_for_title = st.session_state.get("p2_profile", "Selected Profile") # Get from session state or default
    title_p2 = f"Efficient Frontier: {profile_name_for_title} (Full Data Range)"
    if ind_stocks_p2 is not None and len(ind_stocks_p2) == 1: title_p2 = f"Single Stock Performance: {profile_name_for_title} (Full Data Range)"
    _min_var_p2 = min_var_p2 if min_var_p2 and 'Volatility' in min_var_p2 else None
    _max_sharpe_p2 = max_sharpe_p2 if max_sharpe_p2 and 'Volatility' in max_sharpe_p2 else None
    _max_ret_p2 = max_ret_p2 if max_ret_p2 and 'Volatility' in max_ret_p2 else None
    fig_p2 = plot_efficient_frontier_plotly(
        frontier_p2, _min_var_p2, _max_sharpe_p2, _max_ret_p2, ind_stocks_p2, title_p2,
    )
    plot_placeholder_p2.plotly_chart(fig_p2, use_container_width=True)


st.sidebar.header("About")
st.sidebar.info(
    "This app calculates and visualizes the efficient frontier for stock portfolios "
    f"using the full available date range from the local Excel file: '{EXCEL_FILE_PATH}'."
) # Updated sidebar info
st.sidebar.header("Key Points on Frontier")
st.sidebar.markdown(f"- **Min Variance:** Portfolio with the lowest risk.")
st.sidebar.markdown(f"- **Max Sharpe Ratio:** Portfolio with the best risk-adjusted return (risk-free rate: {RISK_FREE_RATE:.0%}).") # Adjusted formatting for 0%
st.sidebar.markdown(f"- **Max Return (on Frontier):** Portfolio on the frontier with the highest expected return.")
st.sidebar.markdown("All portfolio weights are non-negative (no short selling).")

if ALL_STOCK_DATA.empty:
    st.sidebar.error(f"Failed to load data from '{EXCEL_FILE_PATH}'. App functionality is limited.")
elif min_data_date_from_excel and max_data_date_from_excel:
    st.sidebar.info(f"Analysis based on data from: {min_data_date_from_excel.strftime('%Y-%m-%d')} "
                    f"to {max_data_date_from_excel.strftime('%Y-%m-%d')}")