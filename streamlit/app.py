
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import yfinance as yf
from datetime import datetime, timedelta
import json
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import scipy.optimize as sco

# Set page configuration
st.set_page_config(
    page_title="Fintech Portfolio Explorer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E88E5;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #424242;
    }
    .card {
        border-radius: 5px;
        background-color: #f9f9f9;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #1E88E5;
    }
    .metric-label {
        font-size: 1rem;
        color: #616161;
    }
    .positive-change {
        color: #4CAF50;
    }
    .negative-change {
        color: #F44336;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Dashboard", "Portfolio Analysis", "Market Trends", "Predictive Analytics", "Portfolio Optimization", "Anomaly Detection"])

# Sample data for portfolio
@st.cache_data
def load_portfolio_data():
    portfolio = {
        "assets": [
            {"name": "Bitcoin", "ticker": "BTC-USD", "value": 28000, "allocation": 40, "quantity": 0.65},
            {"name": "Ethereum", "ticker": "ETH-USD", "value": 15000, "allocation": 25, "quantity": 6.2},
            {"name": "Solana", "ticker": "SOL-USD", "value": 8000, "allocation": 12, "quantity": 120.5},
            {"name": "Cardano", "ticker": "ADA-USD", "value": 5000, "allocation": 8, "quantity": 12000},
            {"name": "Polkadot", "ticker": "DOT-USD", "value": 4000, "allocation": 7, "quantity": 450},
            {"name": "Avalanche", "ticker": "AVAX-USD", "value": 3000, "allocation": 5, "quantity": 100},
            {"name": "Chainlink", "ticker": "LINK-USD", "value": 2000, "allocation": 3, "quantity": 250}
        ],
        "total_value": 65000
    }
    return portfolio

portfolio = load_portfolio_data()

# Function to get historical data for assets
@st.cache_data
def get_historical_data(tickers, period="1y"):
    try:
        data = {}
        for ticker in tickers:
            stock = yf.Ticker(ticker)
            hist = stock.history(period=period)
            data[ticker] = hist
        return data
    except Exception as e:
        st.error(f"Error fetching historical data: {e}")
        # Fallback to sample data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        
        data = {}
        for ticker in tickers:
            # Generate random price data
            base_price = np.random.uniform(50, 5000)
            # Create a random walk with drift
            random_walk = np.random.normal(0.001, 0.02, size=len(date_range)).cumsum()
            prices = base_price * (1 + random_walk)
            
            hist = pd.DataFrame({
                'Open': prices * np.random.uniform(0.99, 1.0, size=len(date_range)),
                'High': prices * np.random.uniform(1.0, 1.05, size=len(date_range)),
                'Low': prices * np.random.uniform(0.95, 1.0, size=len(date_range)),
                'Close': prices,
                'Volume': np.random.randint(1000000, 10000000, size=len(date_range))
            }, index=date_range)
            
            data[ticker] = hist
        
        return data
    
# Dashboard Page
if page == "Dashboard":
    st.markdown('<p class="main-header">Fintech Portfolio Explorer</p>', unsafe_allow_html=True)
    st.markdown("Track and analyze your cryptocurrency investments with advanced analytics")

    # Key metrics in a row
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<p class="metric-label">Total Portfolio Value</p>', unsafe_allow_html=True)
        st.markdown(f'<p class="metric-value">${portfolio["total_value"]:,}</p>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        # Calculate a sample gain/loss
        gain_percentage = 12.5  # Sample value
        color_class = "positive-change" if gain_percentage > 0 else "negative-change"
        sign = "+" if gain_percentage > 0 else ""
        
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<p class="metric-label">Total Gain/Loss</p>', unsafe_allow_html=True)
        st.markdown(f'<p class="metric-value {color_class}">{sign}{gain_percentage}%</p>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<p class="metric-label">Number of Assets</p>', unsafe_allow_html=True)
        st.markdown(f'<p class="metric-value">{len(portfolio["assets"])}</p>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Asset allocation chart
    st.markdown('<p class="sub-header">Asset Allocation</p>', unsafe_allow_html=True)
    
    df_allocation = pd.DataFrame([
        {"Asset": asset["name"], "Allocation (%)": asset["allocation"]}
        for asset in portfolio["assets"]
    ])
    
    fig_allocation = px.pie(
        df_allocation, 
        values="Allocation (%)", 
        names="Asset", 
        title="Portfolio Allocation",
        color_discrete_sequence=px.colors.qualitative.G10
    )
    fig_allocation.update_traces(textposition='inside', textinfo='percent+label')
    
    st.plotly_chart(fig_allocation, use_container_width=True)
    
    # Asset values
    st.markdown('<p class="sub-header">Asset Values</p>', unsafe_allow_html=True)
    
    df_values = pd.DataFrame([
        {"Asset": asset["name"], "Value (USD)": asset["value"]}
        for asset in portfolio["assets"]
    ])
    
    fig_values = px.bar(
        df_values, 
        x="Asset", 
        y="Value (USD)", 
        title="Asset Values",
        color="Asset",
        color_discrete_sequence=px.colors.qualitative.G10
    )
    
    st.plotly_chart(fig_values, use_container_width=True)
    
    # Asset table
    st.markdown('<p class="sub-header">Portfolio Assets</p>', unsafe_allow_html=True)
    
    df_assets = pd.DataFrame([
        {
            "Asset": asset["name"], 
            "Value (USD)": f"${asset['value']:,}",
            "Allocation (%)": f"{asset['allocation']}%",
            "Quantity": asset["quantity"]
        }
        for asset in portfolio["assets"]
    ])
    
    st.dataframe(df_assets, use_container_width=True)
    
    # Allow downloading of portfolio data
    if st.button("Export Portfolio Data"):
        df_export = pd.DataFrame([
            {
                "Asset": asset["name"], 
                "Ticker": asset["ticker"],
                "Value (USD)": asset["value"],
                "Allocation (%)": asset["allocation"],
                "Quantity": asset["quantity"]
            }
            for asset in portfolio["assets"]
        ])
        
        csv = df_export.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name="portfolio_data.csv",
            mime="text/csv",
        )

# Portfolio Analysis Page
elif page == "Portfolio Analysis":
    st.markdown('<p class="main-header">Portfolio Analysis</p>', unsafe_allow_html=True)
    
    # Get tickers for all assets
    tickers = [asset["ticker"] for asset in portfolio["assets"]]
    
    # Fetch historical data
    historical_data = get_historical_data(tickers)
    
    # Performance over time
    st.markdown('<p class="sub-header">Portfolio Performance</p>', unsafe_allow_html=True)
    
    # Create a combined dataframe of closing prices
    all_close_prices = pd.DataFrame()
    
    for ticker, data in historical_data.items():
        if 'Close' in data.columns:
            all_close_prices[ticker] = data['Close']
    
    # Normalize prices to start at 100 for comparison
    if not all_close_prices.empty:
        normalized_prices = all_close_prices.div(all_close_prices.iloc[0]) * 100
        
        fig_performance = px.line(
            normalized_prices, 
            title="Normalized Asset Performance (Base 100)",
            labels={"value": "Normalized Price", "variable": "Asset"}
        )
        fig_performance.update_layout(showlegend=True)
        
        st.plotly_chart(fig_performance, use_container_width=True)
    
    # Volume analysis
    st.markdown('<p class="sub-header">Trading Volume Analysis</p>', unsafe_allow_html=True)
    
    # Create tabs for each asset
    tabs = st.tabs([asset["name"] for asset in portfolio["assets"]])
    
    for i, tab in enumerate(tabs):
        ticker = portfolio["assets"][i]["ticker"]
        data = historical_data.get(ticker)
        
        if data is not None and 'Volume' in data.columns:
            with tab:
                # Volume chart
                fig_volume = px.bar(
                    data, 
                    y='Volume', 
                    title=f"{portfolio['assets'][i]['name']} Trading Volume"
                )
                st.plotly_chart(fig_volume, use_container_width=True)
                
                # Price with volume overlay
                fig_price_volume = go.Figure()
                
                # Add price line
                fig_price_volume.add_trace(
                    go.Scatter(
                        x=data.index, 
                        y=data['Close'], 
                        name='Price',
                        line=dict(color='blue')
                    )
                )
                
                # Add volume bars with secondary y-axis
                fig_price_volume.add_trace(
                    go.Bar(
                        x=data.index, 
                        y=data['Volume'], 
                        name='Volume',
                        yaxis='y2',
                        opacity=0.3,
                        marker=dict(color='green')
                    )
                )
                
                # Set up dual y-axes
                fig_price_volume.update_layout(
                    title=f"{portfolio['assets'][i]['name']} Price and Volume",
                    yaxis=dict(title="Price"),
                    yaxis2=dict(
                        title="Volume",
                        overlaying="y",
                        side="right"
                    ),
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    )
                )
                
                st.plotly_chart(fig_price_volume, use_container_width=True)
                
                # Volatility analysis
                st.markdown("### Volatility Analysis")
                
                # Calculate daily returns
                data['Daily Return'] = data['Close'].pct_change() * 100
                
                # Display descriptive statistics
                st.write("Daily Returns Statistics:")
                st.write(data['Daily Return'].describe())
                
                # Plot daily returns
                fig_returns = px.histogram(
                    data, 
                    x='Daily Return', 
                    nbins=50,
                    title=f"{portfolio['assets'][i]['name']} Daily Return Distribution"
                )
                st.plotly_chart(fig_returns, use_container_width=True)

# Market Trends Page
elif page == "Market Trends":
    st.markdown('<p class="main-header">Market Trends</p>', unsafe_allow_html=True)
    
    # Get tickers for all assets
    tickers = [asset["ticker"] for asset in portfolio["assets"]]
    
    # Fetch historical data
    historical_data = get_historical_data(tickers)
    
    # Market trends chart
    st.markdown('<p class="sub-header">Market Price Trends</p>', unsafe_allow_html=True)
    
    selected_assets = st.multiselect(
        "Select assets to display",
        [asset["name"] for asset in portfolio["assets"]],
        default=[asset["name"] for asset in portfolio["assets"][:3]]
    )
    
    selected_tickers = [asset["ticker"] for asset in portfolio["assets"] 
                         if asset["name"] in selected_assets]
    
    if selected_tickers:
        all_close_prices = pd.DataFrame()
        
        for ticker in selected_tickers:
            data = historical_data.get(ticker)
            if data is not None and 'Close' in data.columns:
                all_close_prices[ticker] = data['Close']
        
        if not all_close_prices.empty:
            fig_trends = px.line(
                all_close_prices, 
                title="Asset Price Trends",
                labels={"value": "Price (USD)", "variable": "Asset"}
            )
            
            st.plotly_chart(fig_trends, use_container_width=True)
    
    # Correlation analysis
    st.markdown('<p class="sub-header">Asset Correlation Analysis</p>', unsafe_allow_html=True)
    
    # Create a dataframe of returns for correlation analysis
    all_returns = pd.DataFrame()
    
    for ticker, data in historical_data.items():
        if 'Close' in data.columns:
            asset_name = next((asset["name"] for asset in portfolio["assets"] if asset["ticker"] == ticker), ticker)
            all_returns[asset_name] = data['Close'].pct_change().dropna()
    
    if not all_returns.empty:
        # Calculate correlation matrix
        correlation_matrix = all_returns.corr()
        
        # Plot correlation heatmap
        fig_corr = px.imshow(
            correlation_matrix,
            text_auto=True,
            title="Asset Correlation Matrix",
            color_continuous_scale="RdBu_r",
            zmin=-1, 
            zmax=1
        )
        
        st.plotly_chart(fig_corr, use_container_width=True)
        
        st.write("""
        ### Interpreting the Correlation Matrix
        
        - **Close to +1**: Assets move strongly in the same direction
        - **Close to -1**: Assets move strongly in opposite directions
        - **Close to 0**: Little to no relationship between assets' movements
        
        A diversified portfolio typically includes assets with low or negative correlations to reduce overall risk.
        """)

# Predictive Analytics Page
elif page == "Predictive Analytics":
    st.markdown('<p class="main-header">Predictive Analytics</p>', unsafe_allow_html=True)
    
    st.write("""
    This section uses machine learning to predict future asset prices based on historical data.
    Please note that these are educational predictions and should not be used for actual investment decisions.
    """)
    
    # Asset selection for prediction
    selected_asset = st.selectbox(
        "Select an asset to predict",
        [asset["name"] for asset in portfolio["assets"]]
    )
    
    selected_ticker = next((asset["ticker"] for asset in portfolio["assets"] 
                           if asset["name"] == selected_asset), None)
    
    if selected_ticker:
        # Fetch historical data
        historical_data = get_historical_data([selected_ticker])
        data = historical_data.get(selected_ticker)
        
        if data is not None and 'Close' in data.columns:
            # Feature engineering
            df = data.copy()
            
            # Create features (lag prices, moving averages, etc.)
            df['Price_Lag1'] = df['Close'].shift(1)
            df['Price_Lag2'] = df['Close'].shift(2)
            df['Price_Lag3'] = df['Close'].shift(3)
            df['MA5'] = df['Close'].rolling(window=5).mean()
            df['MA10'] = df['Close'].rolling(window=10).mean()
            df['MA20'] = df['Close'].rolling(window=20).mean()
            df['Volatility'] = df['Close'].rolling(window=20).std()
            df['Return'] = df['Close'].pct_change()
            
            # Drop NaN values
            df = df.dropna()
            
            # Prepare features and target
            features = ['Price_Lag1', 'Price_Lag2', 'Price_Lag3', 'MA5', 'MA10', 'MA20', 'Volatility', 'Return']
            X = df[features]
            y = df['Close']
            
            # Split data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)
            
            # Train a RandomForest model
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate error
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            
            # Display prediction results
            st.markdown('<p class="sub-header">Price Prediction Results</p>', unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Root Mean Square Error", f"${rmse:.2f}")
                
            with col2:
                # Calculate prediction performance
                accuracy = 100 * (1 - (rmse / y_test.mean()))
                st.metric("Model Performance", f"{accuracy:.2f}%")
            
            # Plot actual vs predicted prices
            prediction_df = pd.DataFrame({
                'Actual': y_test,
                'Predicted': y_pred
            })
            
            fig_prediction = px.line(
                prediction_df,
                title=f"{selected_asset} Price Prediction",
                labels={"value": "Price (USD)", "variable": "Type"}
            )
            
            st.plotly_chart(fig_prediction, use_container_width=True)
            
            # Feature importance
            st.markdown('<p class="sub-header">Feature Importance</p>', unsafe_allow_html=True)
            
            feature_importance = pd.DataFrame({
                'Feature': features,
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=False)
            
            fig_importance = px.bar(
                feature_importance,
                x='Feature',
                y='Importance',
                title=f"Feature Importance for {selected_asset} Prediction"
            )
            
            st.plotly_chart(fig_importance, use_container_width=True)
            
            # Future prediction
            st.markdown('<p class="sub-header">Future Price Prediction</p>', unsafe_allow_html=True)
            
            prediction_days = st.slider("Number of days to predict", 1, 30, 7)
            
            # Generate future predictions
            future_predictions = []
            last_data = df.iloc[-1:].copy()
            
            for _ in range(prediction_days):
                # Use the model to predict the next day
                X_pred = last_data[features].values
                y_future = model.predict(X_pred)
                
                # Update the prediction with the new value
                future_data = last_data.copy()
                future_data.index = future_data.index + pd.DateOffset(days=1)
                future_data.loc[future_data.index[0], 'Close'] = y_future[0]
                
                # Update features for the next prediction
                future_data['Price_Lag1'] = last_data['Close'].values[0]
                future_data['Price_Lag2'] = last_data['Price_Lag1'].values[0]
                future_data['Price_Lag3'] = last_data['Price_Lag2'].values[0]
                
                # Simple approximation of MA values
                # (In a real implementation, you'd need to keep track of all recent values)
                future_data['MA5'] = (last_data['MA5'].values[0] * 4 + y_future[0]) / 5
                future_data['MA10'] = (last_data['MA10'].values[0] * 9 + y_future[0]) / 10
                future_data['MA20'] = (last_data['MA20'].values[0] * 19 + y_future[0]) / 20
                
                # Approximation of volatility
                future_data['Volatility'] = last_data['Volatility']
                
                # Calculate return
                future_data['Return'] = (y_future[0] / last_data['Close'].values[0]) - 1
                
                # Store prediction
                future_predictions.append({
                    'Date': future_data.index[0],
                    'Price': y_future[0]
                })
                
                # Update last_data for next iteration
                last_data = future_data
            
            # Create DataFrame with predictions
            future_df = pd.DataFrame(future_predictions)
            
            # Combine historical data with predictions for plotting
            combined_df = pd.DataFrame({
                'Date': df.index[-30:].tolist() + future_df['Date'].tolist(),
                'Price': df['Close'][-30:].tolist() + future_df['Price'].tolist(),
                'Type': ['Historical'] * 30 + ['Predicted'] * prediction_days
            })
            
            # Plot the historical data and predictions
            fig_future = px.line(
                combined_df, 
                x='Date', 
                y='Price', 
                color='Type',
                title=f"{selected_asset} Future Price Prediction for Next {prediction_days} Days",
                color_discrete_map={'Historical': 'blue', 'Predicted': 'red'}
            )
            
            st.plotly_chart(fig_future, use_container_width=True)
            
            st.warning("""
            **Disclaimer**: The predictions shown are based on historical patterns and are for educational purposes only.
            Cryptocurrency markets are highly volatile and unpredictable. Do not use these predictions for actual trading decisions.
            """)

# Portfolio Optimization Page
elif page == "Portfolio Optimization":
    st.markdown('<p class="main-header">Portfolio Optimization</p>', unsafe_allow_html=True)
    
    st.write("""
    This section uses Modern Portfolio Theory to optimize asset allocation for the best risk-adjusted returns.
    """)
    
    # Get tickers for all assets
    tickers = [asset["ticker"] for asset in portfolio["assets"]]
    asset_names = [asset["name"] for asset in portfolio["assets"]]
    
    # Fetch historical data
    historical_data = get_historical_data(tickers)
    
    # Create a dataframe of returns for optimization
    all_returns = pd.DataFrame()
    
    for ticker, data in historical_data.items():
        if 'Close' in data.columns:
            asset_name = next((asset["name"] for asset in portfolio["assets"] if asset["ticker"] == ticker), ticker)
            all_returns[asset_name] = data['Close'].pct_change().dropna()
    
    if not all_returns.empty:
        # Calculate mean returns and covariance matrix
        mean_returns = all_returns.mean()
        cov_matrix = all_returns.cov()
        
        # Display current allocation
        st.markdown('<p class="sub-header">Current Allocation</p>', unsafe_allow_html=True)
        
        current_weights = [asset["allocation"] / 100 for asset in portfolio["assets"]]
        current_allocation = pd.DataFrame({
            'Asset': asset_names,
            'Weight': current_weights
        })
        
        fig_current = px.pie(
            current_allocation, 
            values="Weight", 
            names="Asset", 
            title="Current Portfolio Allocation",
            color_discrete_sequence=px.colors.qualitative.G10
        )
        fig_current.update_traces(textposition='inside', textinfo='percent+label')
        
        st.plotly_chart(fig_current, use_container_width=True)
        
        # Portfolio statistics
        st.markdown('<p class="sub-header">Current Portfolio Statistics</p>', unsafe_allow_html=True)
        
        # Calculate portfolio return and risk
        portfolio_return = np.sum(mean_returns * current_weights) * 252  # Annualized return
        portfolio_std_dev = np.sqrt(np.dot(current_weights, np.dot(cov_matrix * 252, current_weights)))  # Annualized volatility
        sharpe_ratio = portfolio_return / portfolio_std_dev  # Sharpe ratio (assuming risk-free rate of 0)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Expected Annual Return", f"{portfolio_return:.2%}")
        
        with col2:
            st.metric("Expected Annual Volatility", f"{portfolio_std_dev:.2%}")
        
        with col3:
            st.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")
        
        # Portfolio optimization
        st.markdown('<p class="sub-header">Portfolio Optimization</p>', unsafe_allow_html=True)
        
        optimization_target = st.radio(
            "Optimization Target",
            ["Maximum Sharpe Ratio", "Minimum Volatility", "Target Return"]
        )
        
        # Function to calculate negative Sharpe ratio
        def neg_sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate=0):
            returns = np.sum(mean_returns * weights) * 252
            std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix * 252, weights)))
            return -(returns - risk_free_rate) / std
        
        # Function to calculate portfolio variance
        def portfolio_volatility(weights, mean_returns, cov_matrix):
            return np.sqrt(np.dot(weights.T, np.dot(cov_matrix * 252, weights)))
        
        # Function to calculate portfolio return
        def portfolio_return(weights, mean_returns):
            return np.sum(mean_returns * weights) * 252
        
        # Constraints
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((0, 1) for _ in range(len(asset_names)))
        
        if optimization_target == "Maximum Sharpe Ratio":
            # Maximize Sharpe ratio
            init_guess = [1/len(asset_names)] * len(asset_names)
            optimized = sco.minimize(neg_sharpe_ratio, init_guess, 
                                    args=(mean_returns, cov_matrix), 
                                    method='SLSQP', 
                                    bounds=bounds, 
                                    constraints=constraints)
            
            optimized_weights = optimized['x']
            
        elif optimization_target == "Minimum Volatility":
            # Minimize volatility
            init_guess = [1/len(asset_names)] * len(asset_names)
            optimized = sco.minimize(portfolio_volatility, init_guess, 
                                    args=(mean_returns, cov_matrix), 
                                    method='SLSQP', 
                                    bounds=bounds, 
                                    constraints=constraints)
            
            optimized_weights = optimized['x']
            
        else:  # Target Return
            target_return = st.slider("Target Annual Return", 
                                    min_value=float(min(mean_returns) * 252), 
                                    max_value=float(max(mean_returns) * 252), 
                                    value=float(portfolio_return))
            
            # Function to minimize volatility subject to target return
            def min_volatility_target_return(weights, mean_returns, cov_matrix, target):
                return portfolio_volatility(weights, mean_returns, cov_matrix)
            
            # Additional constraint for target return
            target_return_constraint = {'type': 'eq', 
                                      'fun': lambda x: portfolio_return(x, mean_returns) - target}
            
            constraints = (constraints, target_return_constraint)
            
            init_guess = [1/len(asset_names)] * len(asset_names)
            optimized = sco.minimize(min_volatility_target_return, init_guess, 
                                    args=(mean_returns, cov_matrix, target_return), 
                                    method='SLSQP', 
                                    bounds=bounds, 
                                    constraints=constraints)
            
            optimized_weights = optimized['x']
        
        # Display optimized allocation
        st.markdown('<p class="sub-header">Optimized Allocation</p>', unsafe_allow_html=True)
        
        optimized_allocation = pd.DataFrame({
            'Asset': asset_names,
            'Weight': optimized_weights
        })
        
        fig_optimized = px.pie(
            optimized_allocation, 
            values="Weight", 
            names="Asset", 
            title="Optimized Portfolio Allocation",
            color_discrete_sequence=px.colors.qualitative.G10
        )
        fig_optimized.update_traces(textposition='inside', textinfo='percent+label')
        
        st.plotly_chart(fig_optimized, use_container_width=True)
        
        # Optimized portfolio statistics
        st.markdown('<p class="sub-header">Optimized Portfolio Statistics</p>', unsafe_allow_html=True)
        
        # Calculate optimized portfolio metrics
        optimized_return = np.sum(mean_returns * optimized_weights) * 252
        optimized_std_dev = np.sqrt(np.dot(optimized_weights, np.dot(cov_matrix * 252, optimized_weights)))
        optimized_sharpe = optimized_return / optimized_std_dev
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Expected Annual Return", f"{optimized_return:.2%}", 
                     f"{optimized_return - portfolio_return:.2%}")
        
        with col2:
            st.metric("Expected Annual Volatility", f"{optimized_std_dev:.2%}", 
                     f"{optimized_std_dev - portfolio_std_dev:.2%}")
        
        with col3:
            st.metric("Sharpe Ratio", f"{optimized_sharpe:.2f}", 
                     f"{optimized_sharpe - sharpe_ratio:.2f}")
        
        # Efficient Frontier
        st.markdown('<p class="sub-header">Efficient Frontier</p>', unsafe_allow_html=True)
        
        # Generate random portfolios
        num_portfolios = 5000
        results = np.zeros((num_portfolios, 3))
        
        for i in range(num_portfolios):
            # Generate random weights
            weights = np.random.random(len(asset_names))
            weights /= np.sum(weights)
            
            # Calculate portfolio metrics
            portfolio_ret = np.sum(mean_returns * weights) * 252
            portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix * 252, weights)))
            sharpe = portfolio_ret / portfolio_vol
            
            results[i, 0] = portfolio_vol
            results[i, 1] = portfolio_ret
            results[i, 2] = sharpe
        
        # Create DataFrame for efficient frontier
        results_df = pd.DataFrame(results, columns=['Volatility', 'Return', 'Sharpe'])
        
        # Plot efficient frontier
        fig_frontier = px.scatter(
            results_df, 
            x='Volatility', 
            y='Return', 
            color='Sharpe',
            title='Efficient Frontier',
            color_continuous_scale='Viridis'
        )
        
        # Add current and optimized portfolios to the plot
        fig_frontier.add_trace(
            go.Scatter(
                x=[portfolio_std_dev], 
                y=[portfolio_return], 
                mode='markers', 
                marker=dict(size=15, color='red'),
                name='Current Portfolio'
            )
        )
        
        fig_frontier.add_trace(
            go.Scatter(
                x=[optimized_std_dev], 
                y=[optimized_return], 
                mode='markers', 
                marker=dict(size=15, color='green'),
                name='Optimized Portfolio'
            )
        )
        
        st.plotly_chart(fig_frontier, use_container_width=True)
        
        st.markdown("""
        ### Understanding the Efficient Frontier
        
        The **Efficient Frontier** represents the set of optimal portfolios that offer the highest expected return for a defined level of risk.
        
        - **Each point** on the scatter plot represents a portfolio with random weights.
        - **Color intensity** shows the Sharpe ratio (risk-adjusted return) of each portfolio.
        - **Current portfolio** is shown in red.
        - **Optimized portfolio** is shown in green.
        
        Portfolios on the upper-left edge of the frontier are considered optimal as they offer the best return for their level of risk.
        """)

# Anomaly Detection Page
elif page == "Anomaly Detection":
    st.markdown('<p class="main-header">Anomaly Detection</p>', unsafe_allow_html=True)
    
    st.write("""
    This section identifies unusual price movements and potential market anomalies using statistical methods.
    """)
    
    # Asset selection for anomaly detection
    selected_asset = st.selectbox(
        "Select an asset to analyze",
        [asset["name"] for asset in portfolio["assets"]]
    )
    
    selected_ticker = next((asset["ticker"] for asset in portfolio["assets"] 
                           if asset["name"] == selected_asset), None)
    
    if selected_ticker:
        # Fetch historical data
        historical_data = get_historical_data([selected_ticker])
        data = historical_data.get(selected_ticker)
        
        if data is not None and 'Close' in data.columns:
            # Prepare data for anomaly detection
            df = data.copy()
            
            # Calculate returns
            df['Return'] = df['Close'].pct_change() * 100
            
            # Calculate rolling statistics
            window_size = st.slider("Moving average window size (days)", 5, 30, 20)
            
            df['MA'] = df['Close'].rolling(window=window_size).mean()
            df['Std'] = df['Close'].rolling(window=window_size).std()
            
            # Calculate z-scores
            df['Z-score'] = (df['Close'] - df['MA']) / df['Std']
            
            # Define anomaly threshold
            threshold = st.slider("Anomaly Z-score threshold", 1.5, 4.0, 3.0)
            
            # Identify anomalies
            df['Anomaly'] = np.where(np.abs(df['Z-score']) > threshold, 1, 0)
            
            # Display results
            st.markdown('<p class="sub-header">Anomaly Detection Results</p>', unsafe_allow_html=True)
            
            # Filter out NaN values
            df_clean = df.dropna()
            
            # Create plot
            fig = go.Figure()
            
            # Add price line
            fig.add_trace(go.Scatter(
                x=df_clean.index,
                y=df_clean['Close'],
                mode='lines',
                name='Price',
                line=dict(color='blue')
            ))
            
            # Add moving average
            fig.add_trace(go.Scatter(
                x=df_clean.index,
                y=df_clean['MA'],
                mode='lines',
                name=f'{window_size}-day MA',
                line=dict(color='green', dash='dash')
            ))
            
            # Add upper and lower bands
            fig.add_trace(go.Scatter(
                x=df_clean.index,
                y=df_clean['MA'] + threshold * df_clean['Std'],
                mode='lines',
                name=f'Upper Band ({threshold}σ)',
                line=dict(color='red', dash='dot'),
                opacity=0.7
            ))
            
            fig.add_trace(go.Scatter(
                x=df_clean.index,
                y=df_clean['MA'] - threshold * df_clean['Std'],
                mode='lines',
                name=f'Lower Band ({threshold}σ)',
                line=dict(color='red', dash='dot'),
                opacity=0.7
            ))
            
            # Add anomaly points
            anomalies = df_clean[df_clean['Anomaly'] == 1]
            
            fig.add_trace(go.Scatter(
                x=anomalies.index,
                y=anomalies['Close'],
                mode='markers',
                name='Anomalies',
                marker=dict(color='red', size=10, symbol='circle')
            ))
            
            fig.update_layout(
                title=f"{selected_asset} Price Anomaly Detection",
                xaxis_title="Date",
                yaxis_title="Price",
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Z-score plot
            fig_zscore = px.line(
                df_clean, 
                x=df_clean.index, 
                y='Z-score', 
                title=f"{selected_asset} Z-Score"
            )
            
            fig_zscore.add_hline(y=threshold, line_dash="dash", line_color="red", annotation_text="Upper Threshold")
            fig_zscore.add_hline(y=-threshold, line_dash="dash", line_color="red", annotation_text="Lower Threshold")
            fig_zscore.add_hline(y=0, line_dash="solid", line_color="green")
            
            # Add anomaly points to z-score plot
            fig_zscore.add_trace(go.Scatter(
                x=anomalies.index,
                y=anomalies['Z-score'],
                mode='markers',
                name='Anomalies',
                marker=dict(color='red', size=10, symbol='circle')
            ))
            
            st.plotly_chart(fig_zscore, use_container_width=True)
            
            # Display anomaly statistics
            st.markdown('<p class="sub-header">Anomaly Statistics</p>', unsafe_allow_html=True)
            
            num_anomalies = len(anomalies)
            anomaly_percentage = (num_anomalies / len(df_clean)) * 100
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Total Anomalies Detected", f"{num_anomalies}")
            
            with col2:
                st.metric("Percentage of Anomalous Days", f"{anomaly_percentage:.2f}%")
            
            # Display anomaly table
            if not anomalies.empty:
                st.markdown('<p class="sub-header">Anomaly Details</p>', unsafe_allow_html=True)
                
                anomaly_table = anomalies[['Close', 'Return', 'Z-score']].copy()
                anomaly_table.index = anomaly_table.index.strftime('%Y-%m-%d')
                anomaly_table.rename(columns={'Close': 'Price', 'Return': 'Daily Return (%)'}, inplace=True)
                
                st.dataframe(anomaly_table, use_container_width=True)
                
                # Download anomaly data
                if st.button("Export Anomaly Data"):
                    csv = anomaly_table.to_csv()
                    st.download_button(
                        label="Download CSV",
                        data=csv,
                        file_name=f"{selected_asset}_anomalies.csv",
                        mime="text/csv",
                    )
                
            else:
                st.info("No anomalies detected with the current threshold.")
            
            st.markdown("""
            ### Understanding Price Anomalies
            
            Price anomalies represent unusual market movements that deviate significantly from expected patterns.
            These anomalies may indicate:
            
            - **Market overreactions** to news or events
            - **Potential manipulation** of asset prices
            - **Liquidity issues** in the market
            - **Technical glitches** in trading systems
            
            Identifying anomalies can help you understand market behavior and potentially adjust your investment strategy accordingly.
            """)

# Add Streamlit app description
st.sidebar.markdown("""
## About This App

This Fintech Portfolio Explorer helps you:

- Track your cryptocurrency investments
- Analyze portfolio performance and risk
- Explore market trends and correlations
- Predict potential future price movements
- Optimize your portfolio allocation
- Detect market anomalies

Built with Streamlit and Python data science libraries.
""")

# Add disclaimer
st.sidebar.info("""
**Disclaimer**: This application is for educational and demonstration purposes only. 
The data, analytics, and predictions should not be considered as financial advice.
""")
