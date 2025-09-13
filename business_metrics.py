"""
Business Metrics Calculation Module for E-commerce Analysis

This module contains functions to calculate various business metrics including
revenue analysis, growth metrics, customer insights, and operational metrics.
All functions are designed to work with configurable date ranges and filters.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')


class BusinessMetrics:
    """
    A class containing methods to calculate various business metrics for e-commerce analysis.
    """
    
    def __init__(self, sales_data: pd.DataFrame):
        """
        Initialize with sales data.
        
        Args:
            sales_data (pd.DataFrame): Complete sales dataset
        """
        self.sales_data = sales_data
        self.setup_plotting_style()
    
    def setup_plotting_style(self):
        """Set up consistent plotting style for all visualizations."""
        plt.style.use('default')
        sns.set_palette("Blues_r")
        
        # Set default figure parameters
        plt.rcParams.update({
            'figure.figsize': (12, 8),
            'axes.titlesize': 14,
            'axes.labelsize': 12,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10
        })
    
    def calculate_revenue_metrics(self, current_period: pd.DataFrame, 
                                comparison_period: pd.DataFrame = None,
                                period_label: str = "Current Period") -> Dict[str, Any]:
        """
        Calculate comprehensive revenue metrics for given period(s).
        
        Args:
            current_period (pd.DataFrame): Sales data for current period
            comparison_period (pd.DataFrame, optional): Sales data for comparison period
            period_label (str): Label for the current period
            
        Returns:
            Dict[str, Any]: Dictionary containing revenue metrics
        """
        metrics = {}
        
        # Current period metrics
        metrics['total_revenue'] = current_period['price'].sum()
        metrics['total_orders'] = current_period['order_id'].nunique()
        metrics['total_items'] = len(current_period)
        metrics['average_order_value'] = current_period.groupby('order_id')['price'].sum().mean()
        metrics['average_item_price'] = current_period['price'].mean()
        metrics['revenue_per_customer'] = (
            current_period.groupby('customer_id')['price'].sum().mean()
            if 'customer_id' in current_period.columns else None
        )
        
        # Comparison metrics if comparison period provided
        if comparison_period is not None and len(comparison_period) > 0:
            comp_revenue = comparison_period['price'].sum()
            comp_orders = comparison_period['order_id'].nunique()
            comp_aov = comparison_period.groupby('order_id')['price'].sum().mean()
            
            metrics['revenue_growth'] = (
                (metrics['total_revenue'] - comp_revenue) / comp_revenue * 100
                if comp_revenue > 0 else 0
            )
            metrics['order_growth'] = (
                (metrics['total_orders'] - comp_orders) / comp_orders * 100
                if comp_orders > 0 else 0
            )
            metrics['aov_growth'] = (
                (metrics['average_order_value'] - comp_aov) / comp_aov * 100
                if comp_aov > 0 else 0
            )
        
        metrics['period_label'] = period_label
        return metrics
    
    def calculate_monthly_growth_trend(self, data: pd.DataFrame, 
                                     year: int = None) -> pd.DataFrame:
        """
        Calculate month-over-month growth trends.
        
        Args:
            data (pd.DataFrame): Sales data
            year (int, optional): Specific year to analyze
            
        Returns:
            pd.DataFrame: Monthly growth trends
        """
        if year:
            data = data[data['year'] == year]
        
        # Group by year and month for revenue calculation
        monthly_revenue = (
            data.groupby(['year', 'month'])['price']
            .sum()
            .reset_index()
        )
        
        # Calculate month-over-month growth
        monthly_revenue['revenue_growth'] = monthly_revenue['price'].pct_change() * 100
        
        # Add month names for better visualization
        month_names = {
            1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
            7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'
        }
        monthly_revenue['month_name'] = monthly_revenue['month'].map(month_names)
        
        return monthly_revenue
    
    def analyze_product_categories(self, data: pd.DataFrame, 
                                 top_n: int = 10) -> pd.DataFrame:
        """
        Analyze product category performance.
        
        Args:
            data (pd.DataFrame): Sales data with product categories
            top_n (int): Number of top categories to return
            
        Returns:
            pd.DataFrame: Product category analysis
        """
        if 'product_category_name' not in data.columns:
            raise ValueError("Product category information not available in dataset")
        
        category_metrics = (
            data.groupby('product_category_name')
            .agg({
                'price': ['sum', 'mean', 'count'],
                'order_id': 'nunique',
                'customer_id': 'nunique' if 'customer_id' in data.columns else lambda x: np.nan
            })
            .round(2)
        )
        
        # Flatten column names
        category_metrics.columns = [
            'total_revenue', 'avg_item_price', 'total_items', 
            'total_orders', 'unique_customers'
        ]
        
        # Calculate additional metrics
        category_metrics['avg_order_value'] = (
            category_metrics['total_revenue'] / category_metrics['total_orders']
        ).round(2)
        
        category_metrics['revenue_share'] = (
            category_metrics['total_revenue'] / category_metrics['total_revenue'].sum() * 100
        ).round(2)
        
        # Sort by total revenue and return top N
        return category_metrics.sort_values('total_revenue', ascending=False).head(top_n)
    
    def analyze_geographic_performance(self, data: pd.DataFrame, 
                                     top_n: int = 15) -> pd.DataFrame:
        """
        Analyze geographic performance by state.
        
        Args:
            data (pd.DataFrame): Sales data with customer state information
            top_n (int): Number of top states to return
            
        Returns:
            pd.DataFrame: Geographic performance analysis
        """
        if 'customer_state' not in data.columns:
            raise ValueError("Customer state information not available in dataset")
        
        state_metrics = (
            data.groupby('customer_state')
            .agg({
                'price': ['sum', 'mean'],
                'order_id': 'nunique',
                'customer_id': 'nunique' if 'customer_id' in data.columns else lambda x: np.nan
            })
            .round(2)
        )
        
        # Flatten column names
        state_metrics.columns = ['total_revenue', 'avg_item_price', 'total_orders', 'unique_customers']
        
        # Calculate additional metrics
        state_metrics['avg_order_value'] = (
            state_metrics['total_revenue'] / state_metrics['total_orders']
        ).round(2)
        
        state_metrics['revenue_share'] = (
            state_metrics['total_revenue'] / state_metrics['total_revenue'].sum() * 100
        ).round(2)
        
        # Sort by total revenue and return top N
        return state_metrics.sort_values('total_revenue', ascending=False).head(top_n)
    
    def analyze_customer_satisfaction(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze customer satisfaction metrics based on review scores.
        
        Args:
            data (pd.DataFrame): Sales data with review scores
            
        Returns:
            Dict[str, Any]: Customer satisfaction metrics
        """
        if 'review_score' not in data.columns:
            return {"error": "Review score information not available in dataset"}
        
        # Remove duplicates to get unique orders for review analysis
        review_data = data[['order_id', 'review_score', 'delivery_days', 'delivery_category']].drop_duplicates()
        
        metrics = {
            'avg_review_score': review_data['review_score'].mean(),
            'total_reviews': len(review_data),
            'review_distribution': review_data['review_score'].value_counts(normalize=True).sort_index(),
            'satisfaction_rate': (review_data['review_score'] >= 4).mean() * 100,  # 4-5 star reviews
            'nps_score': self._calculate_nps(review_data['review_score'])
        }
        
        # Delivery speed vs satisfaction analysis
        if 'delivery_category' in review_data.columns:
            delivery_satisfaction = (
                review_data.groupby('delivery_category')['review_score']
                .agg(['mean', 'count'])
                .round(2)
            )
            metrics['delivery_satisfaction'] = delivery_satisfaction
        
        return metrics
    
    def _calculate_nps(self, scores: pd.Series) -> float:
        """
        Calculate Net Promoter Score from review scores.
        
        Args:
            scores (pd.Series): Review scores
            
        Returns:
            float: NPS score
        """
        promoters = (scores >= 5).sum()
        detractors = (scores <= 2).sum()
        total = len(scores)
        
        return ((promoters - detractors) / total * 100) if total > 0 else 0
    
    def analyze_order_status_distribution(self, orders_data: pd.DataFrame) -> pd.DataFrame:
        """
        Analyze distribution of order statuses.
        
        Args:
            orders_data (pd.DataFrame): Orders dataset
            
        Returns:
            pd.DataFrame: Order status distribution
        """
        status_dist = (
            orders_data['order_status']
            .value_counts(normalize=True)
            .reset_index()
        )
        status_dist.columns = ['order_status', 'percentage']
        status_dist['percentage'] = (status_dist['percentage'] * 100).round(2)
        
        return status_dist
    
    def create_revenue_trend_plot(self, monthly_data: pd.DataFrame, 
                                title: str = "Monthly Revenue Trend") -> plt.Figure:
        """
        Create revenue trend visualization.
        
        Args:
            monthly_data (pd.DataFrame): Monthly aggregated data
            title (str): Plot title
            
        Returns:
            plt.Figure: Matplotlib figure object
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Create x-axis labels
        if 'month_name' in monthly_data.columns:
            x_labels = monthly_data['month_name']
        else:
            x_labels = monthly_data['month']
        
        ax.plot(x_labels, monthly_data['price'], marker='o', linewidth=2, markersize=8)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Month', fontsize=12)
        ax.set_ylabel('Revenue (USD)', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # Format y-axis to show currency
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        return fig
    
    def create_category_performance_plot(self, category_data: pd.DataFrame,
                                       title: str = "Revenue by Product Category") -> plt.Figure:
        """
        Create product category performance visualization.
        
        Args:
            category_data (pd.DataFrame): Category performance data
            title (str): Plot title
            
        Returns:
            plt.Figure: Matplotlib figure object
        """
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Create horizontal bar chart
        y_pos = range(len(category_data))
        bars = ax.barh(y_pos, category_data['total_revenue'])
        
        # Customize the plot
        ax.set_yticks(y_pos)
        ax.set_yticklabels(category_data.index)
        ax.set_xlabel('Total Revenue (USD)', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        # Format x-axis to show currency
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        
        # Add value labels on bars
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax.text(width, bar.get_y() + bar.get_height()/2, 
                   f'${width:,.0f}', ha='left', va='center', fontsize=10)
        
        plt.tight_layout()
        return fig
    
    def create_geographic_heatmap(self, state_data: pd.DataFrame,
                                title: str = "Revenue by State") -> go.Figure:
        """
        Create geographic heatmap using Plotly.
        
        Args:
            state_data (pd.DataFrame): State performance data
            title (str): Plot title
            
        Returns:
            go.Figure: Plotly figure object
        """
        # Reset index to get state as a column
        state_data_plot = state_data.reset_index()
        
        fig = px.choropleth(
            state_data_plot,
            locations='customer_state',
            color='total_revenue',
            locationmode='USA-states',
            scope='usa',
            title=title,
            color_continuous_scale='Blues',
            labels={'total_revenue': 'Total Revenue (USD)'}
        )
        
        fig.update_layout(
            title_font_size=16,
            geo=dict(bgcolor='rgba(0,0,0,0)')
        )
        
        return fig
    
    def create_satisfaction_plot(self, satisfaction_data: Dict[str, Any],
                               title: str = "Customer Satisfaction Analysis") -> plt.Figure:
        """
        Create customer satisfaction visualization.
        
        Args:
            satisfaction_data (Dict): Satisfaction metrics
            title (str): Plot title
            
        Returns:
            plt.Figure: Matplotlib figure object
        """
        if 'error' in satisfaction_data:
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.text(0.5, 0.5, satisfaction_data['error'], 
                   ha='center', va='center', fontsize=14)
            ax.set_title(title)
            return fig
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Review score distribution
        review_dist = satisfaction_data['review_distribution']
        ax1.bar(review_dist.index, review_dist.values * 100)
        ax1.set_title('Review Score Distribution', fontweight='bold')
        ax1.set_xlabel('Review Score')
        ax1.set_ylabel('Percentage (%)')
        ax1.set_ylim(0, max(review_dist.values * 100) * 1.1)
        
        # Add percentage labels on bars
        for i, v in enumerate(review_dist.values * 100):
            ax1.text(review_dist.index[i], v + 1, f'{v:.1f}%', 
                    ha='center', va='bottom', fontweight='bold')
        
        # Delivery satisfaction if available
        if 'delivery_satisfaction' in satisfaction_data:
            delivery_data = satisfaction_data['delivery_satisfaction']
            ax2.bar(range(len(delivery_data)), delivery_data['mean'])
            ax2.set_title('Average Review Score by Delivery Speed', fontweight='bold')
            ax2.set_xlabel('Delivery Category')
            ax2.set_ylabel('Average Review Score')
            ax2.set_xticks(range(len(delivery_data)))
            ax2.set_xticklabels(delivery_data.index, rotation=45)
            ax2.set_ylim(0, 5)
            
            # Add value labels on bars
            for i, v in enumerate(delivery_data['mean']):
                ax2.text(i, v + 0.05, f'{v:.2f}', ha='center', va='bottom', fontweight='bold')
        else:
            ax2.text(0.5, 0.5, 'Delivery data not available', 
                    ha='center', va='center', fontsize=12)
            ax2.set_title('Delivery Analysis Not Available')
        
        plt.tight_layout()
        return fig
    
    def generate_executive_summary(self, current_metrics: Dict[str, Any],
                                 comparison_metrics: Dict[str, Any] = None) -> str:
        """
        Generate executive summary text based on calculated metrics.
        
        Args:
            current_metrics (Dict): Current period metrics
            comparison_metrics (Dict): Comparison period metrics
            
        Returns:
            str: Executive summary text
        """
        summary = f"""
EXECUTIVE SUMMARY - {current_metrics['period_label']}

Key Performance Indicators:
- Total Revenue: ${current_metrics['total_revenue']:,.2f}
- Total Orders: {current_metrics['total_orders']:,}
- Average Order Value: ${current_metrics['average_order_value']:.2f}
- Average Item Price: ${current_metrics['average_item_price']:.2f}
"""
        
        if comparison_metrics and 'revenue_growth' in current_metrics:
            growth_indicator = "📈" if current_metrics['revenue_growth'] > 0 else "📉"
            summary += f"""
Year-over-Year Performance:
- Revenue Growth: {current_metrics['revenue_growth']:.2f}% {growth_indicator}
- Order Growth: {current_metrics['order_growth']:.2f}%
- AOV Growth: {current_metrics['aov_growth']:.2f}%
"""
        
        return summary
    
    def export_metrics_to_csv(self, metrics: Dict[str, Any], filename: str) -> None:
        """
        Export calculated metrics to CSV file.
        
        Args:
            metrics (Dict): Metrics dictionary
            filename (str): Output filename
        """
        # Convert metrics to DataFrame for easy export
        export_data = []
        
        for key, value in metrics.items():
            if isinstance(value, (int, float, str)):
                export_data.append({'metric': key, 'value': value})
            elif isinstance(value, pd.Series):
                for idx, val in value.items():
                    export_data.append({'metric': f'{key}_{idx}', 'value': val})
        
        df_export = pd.DataFrame(export_data)
        df_export.to_csv(filename, index=False)
        print(f"Metrics exported to {filename}")


def calculate_cohort_analysis(sales_data: pd.DataFrame) -> pd.DataFrame:
    """
    Perform cohort analysis to understand customer retention patterns.
    
    Args:
        sales_data (pd.DataFrame): Sales data with customer and date information
        
    Returns:
        pd.DataFrame: Cohort analysis results
    """
    if 'customer_id' not in sales_data.columns:
        raise ValueError("Customer ID information required for cohort analysis")
    
    # Determine customer's first purchase month (cohort)
    customer_cohorts = (
        sales_data.groupby('customer_id')['order_purchase_timestamp']
        .min()
        .reset_index()
    )
    customer_cohorts['cohort_month'] = customer_cohorts['order_purchase_timestamp'].dt.to_period('M')
    
    # Add cohort information back to sales data
    sales_with_cohorts = sales_data.merge(
        customer_cohorts[['customer_id', 'cohort_month']], 
        on='customer_id'
    )
    
    # Calculate period number (months since first purchase)
    sales_with_cohorts['period_number'] = (
        sales_with_cohorts['order_purchase_timestamp'].dt.to_period('M') - 
        sales_with_cohorts['cohort_month']
    ).apply(attrgetter('n'))
    
    # Create cohort table
    cohort_data = (
        sales_with_cohorts.groupby(['cohort_month', 'period_number'])['customer_id']
        .nunique()
        .reset_index()
    )
    
    cohort_sizes = (
        sales_with_cohorts.groupby('cohort_month')['customer_id']
        .nunique()
        .reset_index()
        .rename(columns={'customer_id': 'cohort_size'})
    )
    
    cohort_table = cohort_data.merge(cohort_sizes, on='cohort_month')
    cohort_table['retention_rate'] = cohort_table['customer_id'] / cohort_table['cohort_size']
    
    return cohort_table


def calculate_rfm_analysis(sales_data: pd.DataFrame, 
                          analysis_date: datetime = None) -> pd.DataFrame:
    """
    Perform RFM (Recency, Frequency, Monetary) analysis for customer segmentation.
    
    Args:
        sales_data (pd.DataFrame): Sales data with customer information
        analysis_date (datetime): Reference date for recency calculation
        
    Returns:
        pd.DataFrame: RFM analysis results
    """
    if 'customer_id' not in sales_data.columns:
        raise ValueError("Customer ID information required for RFM analysis")
    
    if analysis_date is None:
        analysis_date = sales_data['order_purchase_timestamp'].max()
    
    # Calculate RFM metrics
    rfm = sales_data.groupby('customer_id').agg({
        'order_purchase_timestamp': lambda x: (analysis_date - x.max()).days,  # Recency
        'order_id': 'nunique',  # Frequency
        'price': 'sum'  # Monetary
    }).reset_index()
    
    rfm.columns = ['customer_id', 'recency', 'frequency', 'monetary']
    
    # Create RFM scores (1-5 scale)
    rfm['r_score'] = pd.qcut(rfm['recency'], 5, labels=[5, 4, 3, 2, 1])  # Lower recency = higher score
    rfm['f_score'] = pd.qcut(rfm['frequency'].rank(method='first'), 5, labels=[1, 2, 3, 4, 5])
    rfm['m_score'] = pd.qcut(rfm['monetary'], 5, labels=[1, 2, 3, 4, 5])
    
    # Combine scores
    rfm['rfm_score'] = rfm['r_score'].astype(str) + rfm['f_score'].astype(str) + rfm['m_score'].astype(str)
    
    return rfm