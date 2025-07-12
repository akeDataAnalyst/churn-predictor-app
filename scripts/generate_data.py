# scripts/generate_data.py
import pandas as pd
import numpy as np
from faker import Faker
from random import randint, choice, uniform
from datetime import datetime, timedelta
from tqdm import tqdm

fake = Faker()
tqdm.pandas()

NUM_CUSTOMERS = 1000
today = datetime.today()

def generate_customers(n):
    customers = []
    for _ in range(n):
        customer_id = fake.uuid4()
        name = fake.name()
        gender = choice(['Male', 'Female'])
        age = randint(18, 70)
        join_date = fake.date_between(start_date='-3y', end_date='-1m')
        location = fake.city()
        email = fake.email()
        phone = fake.phone_number()
        customers.append([customer_id, name, gender, age, join_date, location, email, phone])
    return pd.DataFrame(customers, columns=['customer_id', 'name', 'gender', 'age', 'join_date', 'location', 'email', 'phone'])

def generate_transactions(customers):
    records = []
    for cust_id in customers['customer_id']:
        num_trans = randint(1, 20)
        for _ in range(num_trans):
            date = fake.date_between(start_date='-1y', end_date='today')
            amount = round(uniform(5.0, 500.0), 2)
            product = choice(['SaaS Basic', 'SaaS Pro', 'Consulting', 'Add-on', 'Training'])
            records.append([cust_id, date, amount, product])
    return pd.DataFrame(records, columns=['customer_id', 'transaction_date', 'amount', 'product'])

def generate_support_tickets(customers):
    records = []
    for cust_id in customers['customer_id']:
        num_tickets = randint(0, 5)
        for _ in range(num_tickets):
            date = fake.date_between(start_date='-6m', end_date='today')
            issue_type = choice(['Billing', 'Technical', 'General', 'Account'])
            resolution_days = randint(0, 10)
            records.append([cust_id, date, issue_type, resolution_days])
    return pd.DataFrame(records, columns=['customer_id', 'ticket_date', 'issue_type', 'resolution_time'])

def generate_web_sessions(customers):
    records = []
    for cust_id in customers['customer_id']:
        num_sessions = randint(5, 30)
        for _ in range(num_sessions):
            date = fake.date_between(start_date='-6m', end_date='today')
            duration = round(uniform(2.0, 30.0), 1)
            pages = randint(1, 10)
            device = choice(['Mobile', 'Desktop', 'Tablet'])
            records.append([cust_id, date, duration, pages, device])
    return pd.DataFrame(records, columns=['customer_id', 'session_date', 'duration_min', 'pages_viewed', 'device'])

def generate_churn_labels(customers):
    # Simple rule-based churn: customers with no recent transactions
    labels = []
    for cust_id in customers['customer_id']:
        is_churned = choice([0]*7 + [1]*3)  # ~30% churn rate
        labels.append([cust_id, is_churned])
    return pd.DataFrame(labels, columns=['customer_id', 'churn'])

if __name__ == "__main__":
    customers_df = generate_customers(NUM_CUSTOMERS)
    transactions_df = generate_transactions(customers_df)
    tickets_df = generate_support_tickets(customers_df)
    sessions_df = generate_web_sessions(customers_df)
    churn_df = generate_churn_labels(customers_df)

    # Save to CSV
    customers_df.to_csv("data/raw/customers.csv", index=False)
    transactions_df.to_csv("data/raw/transactions.csv", index=False)
    tickets_df.to_csv("data/raw/support_tickets.csv", index=False)
    sessions_df.to_csv("data/raw/web_sessions.csv", index=False)
    churn_df.to_csv("data/raw/churn_labels.csv", index=False)

    print("✅ All data generated and saved to data/raw/")
