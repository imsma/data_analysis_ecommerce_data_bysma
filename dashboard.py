"""
E-commerce Business Analytics Dashboard

A professional Streamlit dashboard providing comprehensive analysis of e-commerce 
business performance with revenue trends, customer behavior, and operational metrics.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings

# Import custom modules
from data_loader import EcommerceDataLoader
from business_metrics import BusinessMetrics

warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="E-commerce Business Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        margin-bottom: 0.5rem;
    }
    .kpi-card {
        background: white;
        padding: 1.5rem;
        border-radius: 0.75rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border: 1px solid #e5e7eb;
    }
    .kpi-value {
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 0.25rem;
    }
    .kpi-label {
        font-size: 0.9rem;
        color: #6b7280;
        margin-bottom: 0.5rem;
    }
    .trend-positive {
        color: #10b981;
        font-weight: 600;
    }
    .trend-negative {
        color: #ef4444;
        font-weight: 600;
    }
    .bottom-card {
        background: white;
        padding: 2rem;
        border-radius: 0.75rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border: 1px solid #e5e7eb;
        text-align: center;
    }
    .review-score {
        font-size: 3rem;
        font-weight: 700;
        color: #1f77b4;
    }
    .stars {
        font-size: 1.5rem;
        color: #fbbf24;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load and cache the complete dataset"""
    data_loader = EcommerceDataLoader(data_path='ecommerce_data/')
    datasets = data_loader.load_all_datasets()
    complete_sales_data = data_loader.create_complete_sales_dataset(status_filter='delivered')
    return data_loader, datasets, complete_sales_data

def format_currency(value, compact=True):
    """Format currency values with K, M notation"""
    if compact:
        if abs(value) >= 1e6:
            return f"${value/1e6:.1f}M"
        elif abs(value) >= 1e3:
            return f"${value/1e3:.0f}K"
        else:
            return f"${value:.0f}"
    else:
        return f"${value:,.2f}"

def format_number(value, compact=True):
    """Format numbers with K, M notation"""
    if compact:
        if abs(value) >= 1e6:
            return f"{value/1e6:.1f}M"
        elif abs(value) >= 1e3:
            return f"{value/1e3:.1f}K"
        else:
            return f"{value:.0f}"
    else:
        return f"{value:,.0f}"

def get_trend_indicator(current, previous):
    """Get trend indicator with arrow and color"""
    if pd.isna(current) or pd.isna(previous) or previous == 0:
        return "—", "color: #6b7280;"
    
    change = ((current - previous) / previous) * 100
    if change > 0:
        return f"↗ {change:.2f}%", "color: #10b981;"
    elif change < 0:
        return f"↘ {abs(change):.2f}%", "color: #ef4444;"
    else:
        return "— 0.00%", "color: #6b7280;"

def create_revenue_trend_chart(current_data, previous_data, year_filter):
    """Create revenue trend line chart with current and previous period"""
    if len(current_data) == 0:
        # Return empty chart if no data
        fig = go.Figure()
        fig.update_layout(
            title="Monthly Revenue Trend",
            xaxis_title="Month",
            yaxis_title="Revenue",
            height=350,
            plot_bgcolor='white'
        )
        fig.add_annotation(
            text="No data available for selected period",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        return fig
    
    # Current period monthly data
    current_monthly = (
        current_data.groupby(['year', 'month'])['price']
        .sum()
        .reset_index()
    )
    current_monthly['date'] = pd.to_datetime(current_monthly[['year', 'month']].assign(day=1))
    current_monthly = current_monthly.sort_values('date')
    
    # Previous period monthly data
    previous_monthly = pd.DataFrame()
    if len(previous_data) > 0:
        previous_monthly = (
            previous_data.groupby(['year', 'month'])['price']
            .sum()
            .reset_index()
        )
        previous_monthly['date'] = pd.to_datetime(previous_monthly[['year', 'month']].assign(day=1))
        previous_monthly = previous_monthly.sort_values('date')
    
    # Create the plot
    fig = go.Figure()
    
    # Current period (solid line)
    fig.add_trace(go.Scatter(
        x=current_monthly['date'],
        y=current_monthly['price'],
        mode='lines+markers',
        name=f'{year_filter} (Current)',
        line=dict(color='#1f77b4', width=3),
        marker=dict(size=6),
        hovertemplate='<b>%{x|%B %Y}</b><br>Revenue: $%{y:,.0f}<extra></extra>'
    ))
    
    # Previous period (dashed line)
    if len(previous_monthly) > 0:
        # Adjust dates to overlay with current period for comparison
        prev_dates = current_monthly['date'].head(len(previous_monthly))
        fig.add_trace(go.Scatter(
            x=prev_dates,
            y=previous_monthly['price'],
            mode='lines+markers',
            name=f'{year_filter-1} (Previous)',
            line=dict(color='#94a3b8', width=2, dash='dash'),
            marker=dict(size=4)
        ))
    
    fig.update_layout(
        title="Monthly Revenue Trend",
        xaxis_title="Month",
        yaxis_title="Revenue",
        height=350,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        plot_bgcolor='white',
        xaxis=dict(showgrid=True, gridcolor='#f1f5f9'),
        yaxis=dict(
            showgrid=True, 
            gridcolor='#f1f5f9',
            tickformat='$,.0f'
        )
    )
    
    return fig

def create_category_chart(data):
    """Create top 10 categories bar chart with blue gradient"""
    if len(data) == 0 or 'product_category_name' not in data.columns:
        # Return empty chart if no data
        fig = go.Figure()
        fig.update_layout(
            title="Top 10 Product Categories",
            xaxis_title="Revenue",
            height=350,
            plot_bgcolor='white'
        )
        fig.add_annotation(
            text="No category data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        return fig
        
    category_data = (
        data.groupby('product_category_name')['price']
        .sum()
        .reset_index()
        .sort_values('price', ascending=True)  # Ascending for horizontal bar
        .tail(10)  # Top 10
    )
    
    # Create blue gradient colors
    n_categories = len(category_data)
    if n_categories > 1:
        colors = [f'rgba(31, 119, 180, {0.4 + 0.6 * i / (n_categories-1)})' for i in range(n_categories)]
    else:
        colors = ['rgba(31, 119, 180, 1.0)']
    
    fig = go.Figure(go.Bar(
        y=category_data['product_category_name'],
        x=category_data['price'],
        orientation='h',
        marker=dict(color=colors),
        text=[format_currency(val) for val in category_data['price']],
        textposition='outside'
    ))
    
    fig.update_layout(
        title="Top 10 Product Categories",
        xaxis_title="Revenue",
        yaxis_title="",
        height=350,
        plot_bgcolor='white',
        xaxis=dict(showgrid=True, gridcolor='#f1f5f9'),
        yaxis=dict(showgrid=False)
    )
    
    return fig

def create_state_map(data):
    """Create US choropleth map showing revenue by state"""
    if len(data) == 0 or 'customer_state' not in data.columns:
        # Return empty chart if no data
        fig = go.Figure()
        fig.update_layout(
            title="Revenue by State",
            height=350,
            plot_bgcolor='white'
        )
        fig.add_annotation(
            text="No geographic data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        return fig
        
    state_data = (
        data.groupby('customer_state')['price']
        .sum()
        .reset_index()
    )
    
    if len(state_data) == 0:
        fig = go.Figure()
        fig.update_layout(
            title="Revenue by State",
            height=350,
            plot_bgcolor='white'
        )
        fig.add_annotation(
            text="No state data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        return fig
    
    fig = go.Figure(data=go.Choropleth(
        locations=state_data['customer_state'],
        z=state_data['price'],
        locationmode='USA-states',
        colorscale='Blues',
        text=state_data['customer_state'],
        marker_line_color='white',
        colorbar_title="Revenue",
        hovertemplate='<b>%{text}</b><br>Revenue: $%{z:,.0f}<extra></extra>'
    ))
    
    fig.update_layout(
        title='Revenue by State',
        geo=dict(
            scope='usa',
            projection_type='albers usa',
            bgcolor='rgba(0,0,0,0)'
        ),
        height=350,
        plot_bgcolor='white'
    )
    
    return fig

def create_satisfaction_delivery_chart(data):
    """Create satisfaction vs delivery time chart"""
    if len(data) == 0 or 'delivery_category' not in data.columns or 'review_score' not in data.columns:
        # Return empty chart if no data
        fig = go.Figure()
        fig.update_layout(
            title="Average Review Score by Delivery Time",
            xaxis_title="Delivery Time",
            yaxis_title="Average Review Score",
            height=350,
            plot_bgcolor='white'
        )
        fig.add_annotation(
            text="No delivery or review data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        return fig
    
    # Remove duplicates to get unique orders and filter out null values
    review_data = data[['order_id', 'delivery_category', 'review_score']].drop_duplicates()
    review_data = review_data.dropna(subset=['delivery_category', 'review_score'])
    
    if len(review_data) == 0:
        fig = go.Figure()
        fig.update_layout(
            title="Average Review Score by Delivery Time",
            xaxis_title="Delivery Time",
            yaxis_title="Average Review Score",
            height=350,
            plot_bgcolor='white'
        )
        fig.add_annotation(
            text="No valid review data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        return fig
    
    satisfaction_data = (
        review_data.groupby('delivery_category')['review_score']
        .mean()
        .reset_index()
    )
    
    # Define order for delivery categories
    category_order = ['1-3 days', '4-7 days', '8+ days']
    satisfaction_data['delivery_category'] = pd.Categorical(
        satisfaction_data['delivery_category'], 
        categories=category_order, 
        ordered=True
    )
    satisfaction_data = satisfaction_data.sort_values('delivery_category')
    satisfaction_data = satisfaction_data.dropna()  # Remove any NaN categories
    
    fig = go.Figure(go.Bar(
        x=satisfaction_data['delivery_category'],
        y=satisfaction_data['review_score'],
        marker_color='#1f77b4',
        text=[f"{val:.2f}" for val in satisfaction_data['review_score']],
        textposition='outside',
        hovertemplate='<b>%{x}</b><br>Avg Review: %{y:.2f}<extra></extra>'
    ))
    
    fig.update_layout(
        title="Average Review Score by Delivery Time",
        xaxis_title="Delivery Time",
        yaxis_title="Average Review Score",
        height=350,
        plot_bgcolor='white',
        yaxis=dict(range=[0, 5], showgrid=True, gridcolor='#f1f5f9'),
        xaxis=dict(showgrid=False)
    )
    
    return fig

def main():
    # Load data
    data_loader, datasets, complete_sales_data = load_data()
    
    # Header with title and date filters
    col_title, col_filter1, col_filter2 = st.columns([2, 1, 1])
    
    with col_title:
        st.markdown('<h1 class="main-header">E-commerce Business Analytics</h1>', unsafe_allow_html=True)
    
    with col_filter1:
        # Get available years from data - default to 2024 (latest year)
        available_years = sorted(complete_sales_data['year'].unique(), reverse=True)
        year_filter = st.selectbox(
            "Select Year:",
            options=available_years,
            index=0,  # Default to first item (latest year)
            key="year_filter"
        )
    
    with col_filter2:
        # Month filter
        month_options = ["All Months"] + [
            "January", "February", "March", "April", "May", "June",
            "July", "August", "September", "October", "November", "December"
        ]
        month_filter = st.selectbox(
            "Select Month:",
            options=month_options,
            index=0,  # Default to "All Months"
            key="month_filter"
        )
    
    # Filter data based on selected year and month
    current_data = complete_sales_data[complete_sales_data['year'] == year_filter]
    previous_data = complete_sales_data[complete_sales_data['year'] == year_filter - 1]
    
    # Apply month filter if specific month is selected
    if month_filter != "All Months":
        month_number = month_options.index(month_filter)  # Convert month name to number
        current_data = current_data[current_data['month'] == month_number]
        previous_data = previous_data[previous_data['month'] == month_number]
    
    # Show data availability warning if current period has limited data
    period_label = f"{year_filter}" if month_filter == "All Months" else f"{month_filter} {year_filter}"
    if len(current_data) < 10:  # Lower threshold for monthly data
        st.warning(f"⚠️ Very limited data available for {period_label}. Results may not be representative.")
    elif len(current_data) < 100 and month_filter == "All Months":
        st.warning(f"⚠️ Limited data available for {period_label}. Results may not be representative.")
    
    # Show data summary in an expander
    with st.expander("📊 Data Summary", expanded=False):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Records", f"{len(current_data):,}")
        with col2:
            if len(current_data) > 0:
                date_range = f"{current_data['order_purchase_timestamp'].min().strftime('%Y-%m-%d')} to {current_data['order_purchase_timestamp'].max().strftime('%Y-%m-%d')}"
            else:
                date_range = "No data"
            st.metric("Date Range", date_range)
        with col3:
            unique_customers = current_data['customer_id'].nunique() if 'customer_id' in current_data.columns and len(current_data) > 0 else 0
            st.metric("Unique Customers", f"{unique_customers:,}")
    
    # Calculate metrics
    metrics_calc = BusinessMetrics(current_data)
    current_metrics = metrics_calc.calculate_revenue_metrics(current_data, previous_data, f"{year_filter}")
    
    # KPI Row - 4 cards with trend indicators
    st.markdown("<br>", unsafe_allow_html=True)
    kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
    
    # Total Revenue
    with kpi_col1:
        current_revenue = current_metrics['total_revenue']
        prev_revenue = previous_data['price'].sum() if len(previous_data) > 0 else current_revenue
        trend_text, trend_style = get_trend_indicator(current_revenue, prev_revenue)
        
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Total Revenue</div>
            <div class="kpi-value">{format_currency(current_revenue)}</div>
            <div style="{trend_style}">{trend_text}</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Monthly Growth
    with kpi_col2:
        monthly_data = current_data.groupby('month')['price'].sum()
        if len(monthly_data) >= 2:
            latest_month = monthly_data.iloc[-1]
            prev_month = monthly_data.iloc[-2]
            growth_text, growth_style = get_trend_indicator(latest_month, prev_month)
        else:
            growth_text, growth_style = "—", "color: #6b7280;"
        
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Monthly Growth</div>
            <div class="kpi-value">{current_metrics.get('revenue_growth', 0):.2f}%</div>
            <div style="{growth_style}">{growth_text}</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Average Order Value
    with kpi_col3:
        current_aov = current_metrics['average_order_value']
        prev_aov = previous_data.groupby('order_id')['price'].sum().mean() if len(previous_data) > 0 else current_aov
        aov_trend, aov_style = get_trend_indicator(current_aov, prev_aov)
        
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Average Order Value</div>
            <div class="kpi-value">{format_currency(current_aov)}</div>
            <div style="{aov_style}">{aov_trend}</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Total Orders
    with kpi_col4:
        current_orders = current_metrics['total_orders']
        prev_orders = previous_data['order_id'].nunique() if len(previous_data) > 0 else current_orders
        orders_trend, orders_style = get_trend_indicator(current_orders, prev_orders)
        
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Total Orders</div>
            <div class="kpi-value">{format_number(current_orders)}</div>
            <div style="{orders_style}">{orders_trend}</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Charts Grid - 2x2 layout
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    chart_row1_col1, chart_row1_col2 = st.columns(2)
    
    with chart_row1_col1:
        revenue_chart = create_revenue_trend_chart(current_data, previous_data, year_filter)
        st.plotly_chart(revenue_chart, use_container_width=True)
    
    with chart_row1_col2:
        category_chart = create_category_chart(current_data)
        st.plotly_chart(category_chart, use_container_width=True)
    
    chart_row2_col1, chart_row2_col2 = st.columns(2)
    
    with chart_row2_col1:
        if 'customer_state' in current_data.columns:
            state_map = create_state_map(current_data)
            st.plotly_chart(state_map, use_container_width=True)
        else:
            st.info("State data not available")
    
    with chart_row2_col2:
        if 'delivery_category' in current_data.columns and 'review_score' in current_data.columns:
            satisfaction_chart = create_satisfaction_delivery_chart(current_data)
            st.plotly_chart(satisfaction_chart, use_container_width=True)
        else:
            st.info("Delivery or review data not available")
    
    # Bottom Row - 2 cards
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    bottom_col1, bottom_col2 = st.columns(2)
    
    with bottom_col1:
        # Average Delivery Time
        if 'delivery_days' in current_data.columns:
            current_delivery = current_data['delivery_days'].mean()
            prev_delivery = previous_data['delivery_days'].mean() if len(previous_data) > 0 and 'delivery_days' in previous_data.columns else current_delivery
            delivery_trend, delivery_style = get_trend_indicator(current_delivery, prev_delivery)
            
            st.markdown(f"""
            <div class="bottom-card">
                <div class="kpi-label">Average Delivery Time</div>
                <div class="kpi-value">{current_delivery:.1f} days</div>
                <div style="{delivery_style}">{delivery_trend}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="bottom-card">
                <div class="kpi-label">Average Delivery Time</div>
                <div class="kpi-value">N/A</div>
                <div style="color: #6b7280;">Data not available</div>
            </div>
            """, unsafe_allow_html=True)
    
    with bottom_col2:
        # Review Score
        if 'review_score' in current_data.columns:
            avg_review = current_data['review_score'].mean()
            stars = "★" * int(round(avg_review))
            
            st.markdown(f"""
            <div class="bottom-card">
                <div class="kpi-label">Average Review Score</div>
                <div class="review-score">{avg_review:.2f}</div>
                <div class="stars">{stars}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="bottom-card">
                <div class="kpi-label">Average Review Score</div>
                <div class="review-score">N/A</div>
                <div style="color: #6b7280;">Data not available</div>
            </div>
            """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()