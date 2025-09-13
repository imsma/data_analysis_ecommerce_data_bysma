"""
Data Loading and Processing Module for E-commerce Analysis

This module handles loading, cleaning, and initial processing of e-commerce datasets.
It provides functions to load individual datasets and create consolidated sales data
for analysis.
bySMA
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
import warnings

# Suppress pandas warnings for cleaner output
warnings.filterwarnings('ignore', category=pd.errors.SettingWithCopyWarning)


class EcommerceDataLoader:
    """
    A class to handle loading and processing of e-commerce datasets.
    
    This class provides methods to load individual datasets and create
    consolidated views for business analysis.
    """
    
    def __init__(self, data_path: str = 'ecommerce_data/'):
        """
        Initialize the data loader with the path to data files.
        
        Args:
            data_path (str): Path to the directory containing CSV files
        """
        self.data_path = data_path
        self.datasets = {}
        
    def load_orders(self) -> pd.DataFrame:
        """
        Load and process orders dataset.
        
        Returns:
            pd.DataFrame: Processed orders dataset with datetime columns
        """
        orders = pd.read_csv(f'{self.data_path}orders_dataset.csv')
        
        # Convert timestamp columns to datetime
        timestamp_cols = [
            'order_purchase_timestamp',
            'order_approved_at',
            'order_delivered_carrier_date',
            'order_delivered_customer_date',
            'order_estimated_delivery_date'
        ]
        
        for col in timestamp_cols:
            if col in orders.columns:
                orders[col] = pd.to_datetime(orders[col])
        
        # Add derived date columns
        orders['year'] = orders['order_purchase_timestamp'].dt.year
        orders['month'] = orders['order_purchase_timestamp'].dt.month
        orders['quarter'] = orders['order_purchase_timestamp'].dt.quarter
        orders['day_of_week'] = orders['order_purchase_timestamp'].dt.day_name()
        
        self.datasets['orders'] = orders
        return orders
    
    def load_order_items(self) -> pd.DataFrame:
        """
        Load and process order items dataset.
        
        Returns:
            pd.DataFrame: Processed order items dataset
        """
        order_items = pd.read_csv(f'{self.data_path}order_items_dataset.csv')
        
        # Convert shipping limit date to datetime
        if 'shipping_limit_date' in order_items.columns:
            order_items['shipping_limit_date'] = pd.to_datetime(order_items['shipping_limit_date'])
        
        self.datasets['order_items'] = order_items
        return order_items
    
    def load_products(self) -> pd.DataFrame:
        """
        Load and process products dataset.
        
        Returns:
            pd.DataFrame: Processed products dataset
        """
        products = pd.read_csv(f'{self.data_path}products_dataset.csv')
        self.datasets['products'] = products
        return products
    
    def load_customers(self) -> pd.DataFrame:
        """
        Load and process customers dataset.
        
        Returns:
            pd.DataFrame: Processed customers dataset
        """
        customers = pd.read_csv(f'{self.data_path}customers_dataset.csv')
        self.datasets['customers'] = customers
        return customers
    
    def load_reviews(self) -> pd.DataFrame:
        """
        Load and process reviews dataset.
        
        Returns:
            pd.DataFrame: Processed reviews dataset with datetime columns
        """
        reviews = pd.read_csv(f'{self.data_path}order_reviews_dataset.csv')
        
        # Convert timestamp columns to datetime
        timestamp_cols = ['review_creation_date', 'review_answer_timestamp']
        for col in timestamp_cols:
            if col in reviews.columns:
                reviews[col] = pd.to_datetime(reviews[col])
        
        self.datasets['reviews'] = reviews
        return reviews
    
    def load_all_datasets(self) -> Dict[str, pd.DataFrame]:
        """
        Load all datasets and return as a dictionary.
        
        Returns:
            Dict[str, pd.DataFrame]: Dictionary containing all loaded datasets
        """
        self.load_orders()
        self.load_order_items()
        self.load_products()
        self.load_customers()
        self.load_reviews()
        
        return self.datasets
    
    def create_sales_data(self, status_filter: str = 'delivered') -> pd.DataFrame:
        """
        Create consolidated sales dataset by merging orders and order items.
        
        Args:
            status_filter (str): Order status to filter by (default: 'delivered')
            
        Returns:
            pd.DataFrame: Consolidated sales data with relevant columns
        """
        if 'orders' not in self.datasets or 'order_items' not in self.datasets:
            self.load_orders()
            self.load_order_items()
        
        # Select relevant columns for merging
        order_cols = [
            'order_id', 'customer_id', 'order_status', 'order_purchase_timestamp',
            'order_delivered_customer_date', 'year', 'month', 'quarter', 'day_of_week'
        ]
        
        item_cols = [
            'order_id', 'order_item_id', 'product_id', 'price', 'freight_value'
        ]
        
        # Merge datasets
        sales_data = pd.merge(
            left=self.datasets['order_items'][item_cols],
            right=self.datasets['orders'][order_cols],
            on='order_id',
            how='inner'
        )
        
        # Filter by order status if specified
        if status_filter:
            sales_data = sales_data[sales_data['order_status'] == status_filter].copy()
        
        # Calculate total revenue per item (price + freight)
        sales_data['total_revenue'] = sales_data['price'] + sales_data['freight_value']
        
        return sales_data
    
    def create_sales_with_categories(self, status_filter: str = 'delivered') -> pd.DataFrame:
        """
        Create sales data enriched with product category information.
        
        Args:
            status_filter (str): Order status to filter by (default: 'delivered')
            
        Returns:
            pd.DataFrame: Sales data with product categories
        """
        sales_data = self.create_sales_data(status_filter)
        
        if 'products' not in self.datasets:
            self.load_products()
        
        # Merge with product categories
        sales_with_categories = pd.merge(
            left=sales_data,
            right=self.datasets['products'][['product_id', 'product_category_name']],
            on='product_id',
            how='left'
        )
        
        return sales_with_categories
    
    def create_sales_with_geography(self, status_filter: str = 'delivered') -> pd.DataFrame:
        """
        Create sales data enriched with customer geographic information.
        
        Args:
            status_filter (str): Order status to filter by (default: 'delivered')
            
        Returns:
            pd.DataFrame: Sales data with customer geography
        """
        sales_data = self.create_sales_data(status_filter)
        
        if 'customers' not in self.datasets:
            self.load_customers()
        
        # Merge with customer geography
        sales_with_geography = pd.merge(
            left=sales_data,
            right=self.datasets['customers'][['customer_id', 'customer_state', 'customer_city']],
            on='customer_id',
            how='left'
        )
        
        return sales_with_geography
    
    def create_complete_sales_dataset(self, status_filter: str = 'delivered') -> pd.DataFrame:
        """
        Create complete sales dataset with all enrichment data.
        
        Args:
            status_filter (str): Order status to filter by (default: 'delivered')
            
        Returns:
            pd.DataFrame: Complete sales dataset with all dimensions
        """
        # Start with basic sales data
        sales_data = self.create_sales_data(status_filter)
        
        # Load all datasets if not already loaded
        if not all(key in self.datasets for key in ['products', 'customers', 'reviews']):
            self.load_all_datasets()
        
        # Add product information
        sales_complete = pd.merge(
            left=sales_data,
            right=self.datasets['products'][['product_id', 'product_category_name']],
            on='product_id',
            how='left'
        )
        
        # Add customer information
        sales_complete = pd.merge(
            left=sales_complete,
            right=self.datasets['customers'][['customer_id', 'customer_state', 'customer_city']],
            on='customer_id',
            how='left'
        )
        
        # Add review information
        sales_complete = pd.merge(
            left=sales_complete,
            right=self.datasets['reviews'][['order_id', 'review_score']],
            on='order_id',
            how='left'
        )
        
        # Calculate delivery metrics
        if 'order_delivered_customer_date' in sales_complete.columns:
            sales_complete['delivery_days'] = (
                sales_complete['order_delivered_customer_date'] - 
                sales_complete['order_purchase_timestamp']
            ).dt.days
            
            # Categorize delivery speed
            def categorize_delivery_speed(days):
                if pd.isna(days):
                    return 'Unknown'
                elif days <= 3:
                    return '1-3 days'
                elif days <= 7:
                    return '4-7 days'
                else:
                    return '8+ days'
            
            sales_complete['delivery_category'] = sales_complete['delivery_days'].apply(categorize_delivery_speed)
        
        return sales_complete
    
    def filter_by_date_range(self, df: pd.DataFrame, 
                           start_year: int, end_year: int,
                           start_month: Optional[int] = None, 
                           end_month: Optional[int] = None) -> pd.DataFrame:
        """
        Filter dataset by date range.
        
        Args:
            df (pd.DataFrame): Dataset to filter
            start_year (int): Starting year
            end_year (int): Ending year
            start_month (int, optional): Starting month (1-12)
            end_month (int, optional): Ending month (1-12)
            
        Returns:
            pd.DataFrame: Filtered dataset
        """
        # Year filtering
        year_filter = (df['year'] >= start_year) & (df['year'] <= end_year)
        
        # Month filtering (if specified)
        if start_month is not None and end_month is not None:
            if start_year == end_year:
                # Same year, filter by month range
                month_filter = (df['month'] >= start_month) & (df['month'] <= end_month)
                return df[year_filter & month_filter].copy()
            else:
                # Different years, more complex logic needed
                first_year_filter = (df['year'] == start_year) & (df['month'] >= start_month)
                last_year_filter = (df['year'] == end_year) & (df['month'] <= end_month)
                middle_years_filter = (df['year'] > start_year) & (df['year'] < end_year)
                
                return df[first_year_filter | middle_years_filter | last_year_filter].copy()
        
        return df[year_filter].copy()
    
    def get_dataset_info(self) -> Dict[str, Dict]:
        """
        Get summary information about loaded datasets.
        
        Returns:
            Dict[str, Dict]: Dictionary containing dataset information
        """
        info = {}
        
        for name, df in self.datasets.items():
            info[name] = {
                'shape': df.shape,
                'columns': list(df.columns),
                'memory_usage': f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB"
            }
            
        return info


def get_data_dictionary() -> Dict[str, str]:
    """
    Get data dictionary explaining key columns and business terms.
    
    Returns:
        Dict[str, str]: Dictionary mapping column names to descriptions
    """
    return {
        # Orders Dataset
        'order_id': 'Unique identifier for each order',
        'customer_id': 'Unique identifier for each customer',
        'order_status': 'Current status of the order (delivered, canceled, shipped, etc.)',
        'order_purchase_timestamp': 'Date and time when the order was placed',
        'order_approved_at': 'Date and time when the order was approved',
        'order_delivered_customer_date': 'Date and time when order was delivered to customer',
        'order_estimated_delivery_date': 'Estimated delivery date provided to customer',
        
        # Order Items Dataset
        'order_item_id': 'Sequential number identifying items within an order',
        'product_id': 'Unique identifier for each product',
        'seller_id': 'Unique identifier for the seller',
        'price': 'Item price in local currency',
        'freight_value': 'Shipping cost for the item',
        
        # Products Dataset
        'product_category_name': 'Category that the product belongs to',
        'product_name_length': 'Number of characters in product name',
        'product_description_length': 'Number of characters in product description',
        'product_weight_g': 'Product weight in grams',
        
        # Customers Dataset
        'customer_unique_id': 'Unique identifier for customers (can have multiple customer_ids)',
        'customer_zip_code_prefix': 'First 5 digits of customer zip code',
        'customer_city': 'Customer city name',
        'customer_state': 'Customer state abbreviation',
        
        # Reviews Dataset
        'review_id': 'Unique identifier for each review',
        'review_score': 'Rating given by customer (1-5 scale)',
        'review_comment_title': 'Title of the review comment',
        'review_comment_message': 'Review message written by customer',
        'review_creation_date': 'Date when review was created',
        
        # Calculated Metrics
        'total_revenue': 'Sum of item price and freight value',
        'delivery_days': 'Number of days from order to delivery',
        'delivery_category': 'Categorized delivery speed (1-3 days, 4-7 days, 8+ days)',
        'year': 'Year extracted from order purchase timestamp',
        'month': 'Month extracted from order purchase timestamp',
        'quarter': 'Quarter extracted from order purchase timestamp'
    }