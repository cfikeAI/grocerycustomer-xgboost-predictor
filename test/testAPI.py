import requests
import json

data = {
    "age": 20,
    "membership_years": 0.1,
    "number_of_children": 0,
    "quantity": 1,
    "unit_price": 5.0,
    "avg_purchase_value": 5.0,
    "avg_discount_used": 0.9,
    "online_purchases": 0,
    "in_store_purchases": 1,
    "total_sales": 5.0,
    "total_transactions": 1,
    "total_items_purchased": 1,
    "days_since_last_purchase": 365,
    "is_negative_sales": True,

    # Encoded categorical vars (one-hot)
    "gender_Male": False,
    "gender_Other": False,

    "income_bracket_Low": True,
    "income_bracket_Medium": False,

    "marital_status_Married": False,
    "marital_status_Single": True,

    "education_level_High School": True,
    "education_level_Master's": False,
    "education_level_PhD": False,

    "occupation_Retired": False,
    "occupation_Self-Employed": False,
    "occupation_Student": True,
    "occupation_Unemployed": False,

    "product_category_Electronics": False,
    "product_category_Groceries": False,
    "product_category_Home Goods": False,
    "product_category_Toys": True,

    "purchase_frequency_Monthly": False,
    "purchase_frequency_Unknown": True,
    "purchase_frequency_Weekly": False,
    "purchase_frequency_Yearly": False,

    "promotion_type_Buy One Get One Free": False,
    "promotion_type_None": False,
    "promotion_type_Seasonal Discount": True,

    "loyalty_program_Yes": False,
    "promotion_effectiveness_Low": True,
    "promotion_effectiveness_Medium": False
}

response = requests.post("http://127.0.0.1:8000/predict", json=data)
print(response.json())
