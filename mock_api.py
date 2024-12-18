from fastapi import FastAPI
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware
from fuzzywuzzy import process
from loguru import logger
from ast import literal_eval
import numpy as np
from embedding import search_embeddings
import boto3


# Load Orders dataset
DATASET_PATH = "./Data/Order_Data_Dataset.csv"
df = pd.read_csv(DATASET_PATH)

# Load Product Information dataset
s3 = boto3.client("s3")
def download_file_from_s3(bucket_name, object_name, file_name):
    s3.download_file(bucket_name, object_name, file_name)
bucket_name = "chatbot-store-genailabs"
object_name = "Product_Information_Dataset_with_embeddings.csv"
file_name = "./Data/Product_Information_Dataset_with_embeddings.csv"  # local file name
download_file_from_s3(bucket_name, object_name, file_name)
PRODUCT_PATH = "./Data/Product_Information_Dataset_with_embeddings.csv"
df_product = pd.read_csv(PRODUCT_PATH)
df_product["embedding"] = df_product.embedding.apply(literal_eval).apply(np.array)


# Initialize FastAPI app
app = FastAPI(title="E-commerce Dataset API", description="API for querying e-commerce sales data")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this to your needs
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Clean data (e.g., handle NaN values) at the start
df.fillna(value="", inplace=True)


@app.get("/health")
def health():
    return {"status": "ok"}

# Endpoint to get all data
@app.get("/data")
def get_all_data():
    """Retrieve all records in the dataset."""
    return df.to_dict(orient="records")

# Endpoint to filter data by Customer ID
@app.get("/data/customer/{customer_id}")
def get_customer_data(customer_id: int):
    """Retrieve all records for a specific Customer ID."""
    filtered_data = df[df["Customer_Id"] == customer_id]
    if filtered_data.empty:
        return {"error": f"No data found for Customer ID {customer_id}"}
    return filtered_data.to_dict(orient="records")

# Endpoint to filter data by Product Category
@app.get("/data/product-category/{category}")
def get_product_category_data(category: str):
    """Retrieve all records for a specific Product Category."""
    filtered_data = df[df["Product_Category"].str.contains(category, case=False, na=False)]
    if filtered_data.empty:
        return {"error": f"No data found for Product Category '{category}'"}
    return filtered_data.to_dict(orient="records")

# Endpoint to get orders with specific priorities
@app.get("/data/order-priority/{priority}")
def get_orders_by_priority(priority: str):
    """Retrieve all orders with the given priority."""
    filtered_data = df[df["Order_Priority"].str.contains(priority, case=False, na=False)]
    if filtered_data.empty:
        return {"error": f"No data found for Order Priority '{priority}'"}
    return filtered_data.to_dict(orient="records")

# Endpoint to calculate total sales by Product Category
@app.get("/data/total-sales-by-category")
def total_sales_by_category():
    """Calculate total sales by Product Category."""
    sales_summary = df.groupby("Product_Category")["Sales"].sum().reset_index()
    return sales_summary.to_dict(orient="records")

# Endpoint to get high-profit products
@app.get("/data/high-profit-products")
def high_profit_products(min_profit: float = 100.0):
    """Retrieve products with profit greater than the specified value."""
    filtered_data = df[df["Profit"] > min_profit]
    if filtered_data.empty:
        return {"error": f"No products found with profit greater than {min_profit}"}
    return filtered_data.to_dict(orient="records")

# Endpoint to get shipping cost summary
@app.get("/data/shipping-cost-summary")
def shipping_cost_summary():
    """Retrieve the average, minimum, and maximum shipping cost."""
    summary = {
        "average_shipping_cost": df["Shipping_Cost"].mean(),
        "min_shipping_cost": df["Shipping_Cost"].min(),
        "max_shipping_cost": df["Shipping_Cost"].max()
    }
    return summary

# Endpoint to calculate total profit by Gender
@app.get("/data/profit-by-gender")
def profit_by_gender():
    """Calculate total profit by customer gender."""
    profit_summary = df.groupby("Gender")["Profit"].sum().reset_index()
    return profit_summary.to_dict(orient="records")


# Product Information

# Endpoint to search for products based on a query
@app.get("/data/search-products")
def search_products_embedding(query: str, sort_column: str = "average_rating", sort_order: str = "desc", limit: int = 5):
    logger.info(
        f"Searching for products with query: {query}, sort_column: {sort_column}, sort_order: {sort_order}, limit: {limit}"
    )
    result = search_embeddings(df_product, query, n=10000000)

    # Limit the results
    result = result.head(limit)

    # Sort the results based on the specified column and order
    result.sort_values(by=sort_column, ascending=(sort_order == "asc"), inplace=True)

    # Fill NaN values to avoid JSON serialization issues
    result = result.fillna("")
    
    return result.to_dict(orient="records")


def fuzzy_search(df, query, column, limit=10):
    # Extract the column values as a list
    choices = df[column].tolist()

    # Perform fuzzy matching
    results = process.extract(query, choices, limit=limit)

    # Extract the matched rows from the DataFrame
    matched_indices = [choices.index(result[0]) for result in results]
    matched_df = df.iloc[matched_indices].copy()

    # Add the score to the matched DataFrame
    matched_df["score"] = [result[1] for result in results]

    return matched_df

# Deprecated. Use /data/search-products instead.
@app.get("/data/search-products-fuzzy")
def search_products_fuzzy(query: str, sort_column: str = "average_rating", sort_order: str = "desc", limit: int = 5):
    logger.info(f"Searching for products with query: {query}, sort_column: {sort_column}, sort_order: {sort_order}, limit: {limit}")
    # Perform fuzzy search on multiple columns
    columns_to_search = [
        "title",
        "description",
        # "main_category",
        # "features",
        # "categories",
        # "details",
    ]
    combined_matches = pd.DataFrame()

    for column in columns_to_search:
        matches = fuzzy_search(
            df_product, query, column, limit=100000000
        )  # a large number to get all matches
        # keep the highest score if there are duplicates, because fuzzy searching on one column can have a different score than another column
        combined_matches = pd.concat([combined_matches, matches]).sort_values(by='score',ascending=False).drop_duplicates(subset=['title'], keep='first')
    
    # Limit the results
    combined_matches = combined_matches.head(limit)

    # Sort the combined matches based on the specified column and order
    top_results = combined_matches.sort_values(by=sort_column, ascending=(sort_order == "asc"))
    # Fill NaN values to avoid JSON serialization issues
    top_results = top_results.fillna("")

    return top_results.to_dict(orient="records")


# Endpoint to get unique column names
@app.get("/data/product-columns")
def get_product_columns():
    """Retrieve column names in the products dataset."""
    logger.info("Retrieving column names in the products dataset")
    product_columns = df_product.columns.tolist()
    return {"product_columns": product_columns}
