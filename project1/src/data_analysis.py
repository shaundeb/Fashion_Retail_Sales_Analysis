# src/data_analysis.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Configuración de estilo
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class FashionRetailAnalysis:
    def __init__(self, data_path):
        self.df = pd.read_csv(data_path)
        self.df['date_purchase'] = pd.to_datetime(self.df['date_purchase'])
        
    def basic_info(self):
        """Información básica del dataset"""
        print("="*50)
        print("INFORMACIÓN BÁSICA DEL DATASET")
        print("="*50)
        print(f"Total de registros: {len(self.df):,}")
        print(f"Período de datos: {self.df['date_purchase'].min()} to {self.df['date_purchase'].max()}")
        print(f"Total de clientes únicos: {self.df['customer_id'].nunique():,}")
        print(f"Total de productos únicos: {self.df['item_purchased'].nunique():,}")
        print(f"Rango de precios: ${self.df['purchase_amount_usd'].min():.2f} - ${self.df['purchase_amount_usd'].max():.2f}")
        
    def sales_trend_analysis(self):
        """Análisis de tendencias de ventas"""
        print("\n" + "="*50)
        print("ANÁLISIS DE TENDENCIAS DE VENTAS")
        print("="*50)
        
        # Ventas por mes
        monthly_sales = self.df.groupby(['purchase_year', 'purchase_month']).agg({
            'purchase_amount_usd': ['sum', 'count', 'mean']
        }).round(2)
        monthly_sales.columns = ['total_sales', 'transaction_count', 'avg_transaction']
        
        # Ventas por trimestre
        quarterly_sales = self.df.groupby('purchase_quarter').agg({
            'purchase_amount_usd': ['sum', 'count', 'mean']
        }).round(2)
        quarterly_sales.columns = ['total_sales', 'transaction_count', 'avg_transaction']
        
        print("Ventas mensuales:")
        print(monthly_sales)
        print("\nVentas trimestrales:")
        print(quarterly_sales)
        
        return monthly_sales, quarterly_sales
    
    def customer_analysis(self):
        """Análisis de comportamiento del cliente"""
        print("\n" + "="*50)
        print("ANÁLISIS DE CLIENTES")
        print("="*50)
        
        # Frecuencia de compra por cliente
        customer_frequency = self.df.groupby('customer_id').agg({
            'purchase_amount_usd': ['count', 'sum', 'mean']
        }).round(2)
        customer_frequency.columns = ['purchase_count', 'total_spent', 'avg_spent']
        
        # Segmentación de clientes
        customer_segments = pd.cut(customer_frequency['total_spent'], 
                                  bins=[0, 100, 500, 1000, float('inf')],
                                  labels=['Low', 'Medium', 'High', 'VIP'])
        
        print("Estadísticas de clientes:")
        print(f"Clientes con una compra: {(customer_frequency['purchase_count'] == 1).sum()}")
        print(f"Clientes recurrentes: {(customer_frequency['purchase_count'] > 1).sum()}")
        print("\nSegmentación de clientes:")
        print(customer_segments.value_counts())
        
        return customer_frequency, customer_segments
    
    def product_analysis(self):
        """Análisis de productos"""
        print("\n" + "="*50)
        print("ANÁLISIS DE PRODUCTOS")
        print("="*50)
        
        # Productos más vendidos
        top_products = self.df['item_purchased'].value_counts().head(10)
        
        # Productos con mayor revenue
        product_revenue = self.df.groupby('item_purchased')['purchase_amount_usd'].agg(['sum', 'count', 'mean']).round(2)
        product_revenue.columns = ['total_revenue', 'units_sold', 'avg_price']
        top_revenue_products = product_revenue.sort_values('total_revenue', ascending=False).head(10)
        
        print("Top 10 productos más vendidos:")
        print(top_products)
        print("\nTop 10 productos por revenue:")
        print(top_revenue_products)
        
        return top_products, top_revenue_products
    
    def payment_method_analysis(self):
        """Análisis de métodos de pago"""
        print("\n" + "="*50)
        print("ANÁLISIS DE MÉTODOS DE PAGO")
        print("="*50)
        
        payment_stats = self.df.groupby('payment_method').agg({
            'purchase_amount_usd': ['count', 'sum', 'mean'],
            'review_rating': 'mean'
        }).round(2)
        payment_stats.columns = ['transaction_count', 'total_amount', 'avg_amount', 'avg_rating']
        
        print("Estadísticas por método de pago:")
        print(payment_stats)
        
        return payment_stats
    
    def review_analysis(self):
        """Análisis de reseñas"""
        print("\n" + "="*50)
        print("ANÁLISIS DE RESEÑAS")
        print("="*50)
        
        review_stats = self.df['review_rating'].describe()
        rating_distribution = self.df['review_rating'].value_counts().sort_index()
        
        # Correlación entre rating y monto de compra
        correlation = self.df[['review_rating', 'purchase_amount_usd']].corr().iloc[0,1]
        
        print("Estadísticas de reseñas:")
        print(review_stats)
        print(f"\nCorrelación rating-monto: {correlation:.3f}")
        
        return review_stats, rating_distribution
    
    def seasonal_analysis(self):
        """Análisis estacional"""
        print("\n" + "="*50)
        print("ANÁLISIS ESTACIONAL")
        print("="*50)
        
        # Ventas por día de la semana
        self.df['day_of_week'] = self.df['date_purchase'].dt.day_name()
        self.df['month_name'] = self.df['date_purchase'].dt.month_name()
        
        daily_sales = self.df.groupby('day_of_week')['purchase_amount_usd'].agg(['sum', 'count']).round(2)
        monthly_sales = self.df.groupby('month_name')['purchase_amount_usd'].agg(['sum', 'count']).round(2)
        
        print("Ventas por día de la semana:")
        print(daily_sales)
        print("\nVentas por mes:")
        print(monthly_sales)
        
        return daily_sales, monthly_sales
    
    def generate_dashboard(self):
        """Generar dashboard visual"""
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        fig.suptitle('Fashion Retail Analytics Dashboard', fontsize=16, fontweight='bold')
        
        # 1. Tendencia de ventas mensuales
        monthly_trend = self.df.groupby(['purchase_year', 'purchase_month'])['purchase_amount_usd'].sum().reset_index()
        monthly_trend['period'] = monthly_trend['purchase_year'].astype(str) + '-' + monthly_trend['purchase_month'].astype(str).str.zfill(2)
        axes[0,0].plot(monthly_trend['period'], monthly_trend['purchase_amount_usd'], marker='o')
        axes[0,0].set_title('Tendencia de Ventas Mensuales')
        axes[0,0].tick_params(axis='x', rotation=45)
        
        # 2. Distribución de métodos de pago
        payment_dist = self.df['payment_method'].value_counts()
        axes[0,1].pie(payment_dist.values, labels=payment_dist.index, autopct='%1.1f%%')
        axes[0,1].set_title('Distribución de Métodos de Pago')
        
        # 3. Top 10 productos por revenue
        top_products = self.df.groupby('item_purchased')['purchase_amount_usd'].sum().nlargest(10)
        axes[1,0].barh(range(len(top_products)), top_products.values)
        axes[1,0].set_yticks(range(len(top_products)))
        axes[1,0].set_yticklabels(top_products.index)
        axes[1,0].set_title('Top 10 Productos por Revenue')
        
        # 4. Distribución de ratings
        rating_dist = self.df['review_rating'].value_counts().sort_index()
        axes[1,1].bar(rating_dist.index, rating_dist.values)
        axes[1,1].set_title('Distribución de Ratings')
        axes[1,1].set_xlabel('Rating')
        axes[1,1].set_ylabel('Frecuencia')
        
        # 5. Ventas por día de la semana
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        daily_sales = self.df.groupby('day_of_week')['purchase_amount_usd'].sum().reindex(day_order)
        axes[2,0].bar(range(len(daily_sales)), daily_sales.values)
        axes[2,0].set_xticks(range(len(daily_sales)))
        axes[2,0].set_xticklabels(day_order, rotation=45)
        axes[2,0].set_title('Ventas por Día de la Semana')
        
        # 6. Boxplot de precios
        axes[2,1].boxplot(self.df['purchase_amount_usd'])
        axes[2,1].set_title('Distribución de Precios')
        axes[2,1].set_ylabel('USD')
        
        plt.tight_layout()
        plt.savefig('retail_analytics_dashboard.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_report(self):
        """Generar reporte completo"""
        self.basic_info()
        self.sales_trend_analysis()
        self.customer_analysis()
        self.product_analysis()
        self.payment_method_analysis()
        self.review_analysis()
        self.seasonal_analysis()
        self.generate_dashboard()
        
        # Guardar métricas clave
        key_metrics = {
            'total_revenue': self.df['purchase_amount_usd'].sum(),
            'avg_transaction_value': self.df['purchase_amount_usd'].mean(),
            'total_customers': self.df['customer_id'].nunique(),
            'avg_rating': self.df['review_rating'].mean(),
            'conversion_rate': len(self.df) / self.df['customer_id'].nunique()
        }
        
        print("\n" + "="*50)
        print("MÉTRICAS CLAVE")
        print("="*50)
        for metric, value in key_metrics.items():
            print(f"{metric}: {value:,.2f}")