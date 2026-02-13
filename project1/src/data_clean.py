# src/data_cleaning.py
import pandas as pd

def clean_data(input_path, output_path):
    """
    Limpia un dataset de ventas de fashion retail y lo guarda en un CSV limpio.
    
    Parámetros:
    - input_path: ruta del CSV de entrada
    - output_path: ruta donde guardar el CSV limpio
    """
    # Leer CSV
    df = pd.read_csv(input_path)

    # Renombrar columnas
    df.columns = ["customer_id", "item_purchased", "purchase_amount_usd",
                  "date_purchase", "review_rating", "payment_method"]

    # Eliminar duplicados
    df = df.drop_duplicates()

    # Convertir tipos
    df["date_purchase"] = pd.to_datetime(df["date_purchase"], dayfirst=True, errors="coerce")
    df["purchase_amount_usd"] = pd.to_numeric(df["purchase_amount_usd"], errors="coerce")
    df["review_rating"] = pd.to_numeric(df["review_rating"], errors="coerce")
    df["payment_method"] = df["payment_method"].str.lower().str.strip()

    # Eliminar filas sin datos esenciales
    df = df.dropna(subset=["purchase_amount_usd", "item_purchased", "date_purchase"])

    # Crear columnas de año, mes y trimestre
    df["purchase_year"] = df["date_purchase"].dt.year
    df["purchase_month"] = df["date_purchase"].dt.month
    df["purchase_quarter"] = df["date_purchase"].dt.to_period("Q")

    # Guardar CSV limpio
    df.to_csv(output_path, index=False)

    return df
