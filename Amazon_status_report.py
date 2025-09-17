# Import
import pandas as pd
import streamlit as st
import os
import joblib
import sklearn

# Load Pipeline
model_path = "Status_pipeline.pkl"
if os.path.exists(model_path):
    pipeline = joblib.load(model_path)
    st.success("MODEL LOADED SUCCESSFULLY")
else:
    st.warning(f"Model file {model_path} does not exist")

# --- Amazon Logo + Title (Centered) ---
st.markdown(
    """
    <div style="text-align: center;">
        <img src="https://upload.wikimedia.org/wikipedia/commons/a/a9/Amazon_logo.svg" width="200">
        <h1>📦 Amazon Delivery Tracker</h1>
        <h3 style="color: gray;">Stay updated with real-time order status and shipment progress</h3>
    </div>
    """,
    unsafe_allow_html=True
)

st.write("Fill your details below to check the delivery status:")

# Form Inputs
with st.form("Status_Report"):
    Product = st.selectbox("Product", ["Book", "Headphones", "Jeans", "Laptop", "Refrigerator", "Running Shoes", "Smartphone", "Smartwatch", "T-Shirt", "Washing Machine"])
    Category = st.selectbox("Category", ["Books", "Clothing", "Electronics", "Footwear", "Home Appliances"])
    Price = st.number_input("Price (In INR)", min_value=15, max_value=1200)
    Quantity = st.number_input("Quantity", min_value=1, max_value=5)
    Total_Sales = st.number_input("Total Sales", min_value=15, max_value=6000)
    Customer_Location = st.selectbox("Customer Location", ["Boston", "Chicago", "Dallas", "Denver", "Houston", "Los Angeles", "Miami", "New York", "San Francisco", "Seattle"])
    Payment_Method = st.selectbox("Payment Method", ["Amazon Pay", "Credit Card", "Debit Card", "Gift Card", "Paypal"])
    
    Submitted = st.form_submit_button("Check Delivery Status")

# Prediction and Output
if Submitted:
    input_data = pd.DataFrame([{
        "Product": Product,
        "Category": Category,
        "Price": Price,
        "Quantity": Quantity,
        "Total Sales": Total_Sales,
        "Customer Location": Customer_Location,
        "Payment Method": Payment_Method,
    }])

    # Reverse Mapping (same used in training with .map)
    reverse_map = {0: "Cancelled", 1: "Completed", 2: "Pending"}

    # Get prediction
    prediction_num = pipeline.predict(input_data)[0]
    prediction = reverse_map[prediction_num]

    # Badge Style Output
    status_styles = {
        "Completed": {"color": "#4CAF50", "emoji": "✅", "text": "Order Completed Successfully"},
        "Pending": {"color": "#2196F3", "emoji": "⏳", "text": "Order Pending"},
        "Cancelled": {"color": "#F44336", "emoji": "❌", "text": "Order Cancelled"}
    }

    status_info = status_styles.get(prediction, {"color": "#999999", "emoji": "📦", "text": "Status Unknown"})

    st.markdown(
        f"""
        <div style="
            text-align: center;
            margin-top: 20px;
        ">
            <div style="
                display:inline-block;
                padding: 12px 20px;
                border-radius: 12px;
                background-color: {status_info['color']};
                color: white;
                font-size: 20px;
                font-weight: bold;">
                {status_info['emoji']} {status_info['text']}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
