"""Generate a realistic, deliberately messy synthetic e-commerce dataset.

Outputs raw CSVs to data/raw/: customers, products, locations, orders, order_items.
Seeded, so results are reproducible.
"""
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
rng = np.random.default_rng(SEED)
OUT = Path(__file__).resolve().parent.parent / "data" / "raw"
OUT.mkdir(parents=True, exist_ok=True)

N_CUSTOMERS = 6000
START, END = pd.Timestamp("2023-01-01"), pd.Timestamp("2025-09-30")

# ---------- locations (Indian cities) ----------
locations = pd.DataFrame(
    [
        ("Bengaluru", "Karnataka", "South", 1.00, 12.97, 77.59),
        ("Mumbai", "Maharashtra", "West", 1.05, 19.08, 72.88),
        ("Delhi", "Delhi", "North", 1.00, 28.61, 77.21),
        ("Hyderabad", "Telangana", "South", 0.80, 17.39, 78.49),
        ("Chennai", "Tamil Nadu", "South", 0.75, 13.08, 80.27),
        ("Pune", "Maharashtra", "West", 0.70, 18.52, 73.86),
        ("Kolkata", "West Bengal", "East", 0.55, 22.57, 88.36),
        ("Ahmedabad", "Gujarat", "West", 0.50, 23.02, 72.57),
        ("Jaipur", "Rajasthan", "North", 0.35, 26.91, 75.79),
        ("Lucknow", "Uttar Pradesh", "North", 0.30, 26.85, 80.95),
        ("Kochi", "Kerala", "South", 0.30, 9.93, 76.27),
        ("Patna", "Bihar", "East", 0.18, 25.59, 85.14),
        ("Guwahati", "Assam", "North-East", 0.12, 26.14, 91.74),
        ("Bhubaneswar", "Odisha", "East", 0.15, 20.30, 85.82),
    ],
    columns=["city", "state", "region", "weight", "lat", "lon"],
)
locations.insert(0, "location_id", [f"L{i:02d}" for i in range(1, len(locations) + 1)])

# ---------- products ----------
cats = {
    # category: (subcats, price range, margin range)
    "Electronics": (["Headphones", "Smartwatch", "Power Bank", "Speaker", "Webcam"], (800, 9000), (0.10, 0.22)),
    "Fashion": (["T-Shirt", "Jeans", "Sneakers", "Kurta", "Jacket"], (400, 3500), (0.35, 0.55)),
    "Home & Kitchen": (["Cookware", "Bedsheet", "Lamp", "Organizer", "Mixer"], (300, 6000), (0.25, 0.40)),
    "Beauty": (["Skincare", "Haircare", "Fragrance", "Makeup"], (200, 2500), (0.45, 0.65)),
    "Books": (["Fiction", "Self-help", "Academic", "Comics"], (150, 900), (0.20, 0.30)),
    "Sports": (["Yoga Mat", "Dumbbells", "Cricket Bat", "Running Shoes"], (400, 5000), (0.20, 0.35)),
    "Toys": (["Board Game", "Puzzle", "RC Car", "Soft Toy"], (250, 3000), (0.28, 0.45)),
}
prod_rows = []
pid = 1
for cat, (subs, (lo, hi), (mlo, mhi)) in cats.items():
    for _ in range(14):
        price = round(float(np.exp(rng.uniform(np.log(lo), np.log(hi)))) / 10) * 10
        margin = rng.uniform(mlo, mhi)
        prod_rows.append(
            (f"P{pid:03d}", f"{cat[:4].upper()}-{pid:03d}", cat, rng.choice(subs), price, round(price * (1 - margin), 2))
        )
        pid += 1
products = pd.DataFrame(
    prod_rows, columns=["product_id", "product_name", "category", "sub_category", "unit_price", "unit_cost"]
)
# popularity (long tail) and a couple of deliberately weak categories
products["pop"] = rng.pareto(1.6, len(products)) + 0.3
products.loc[products.category.isin(["Books", "Toys"]), "pop"] *= 0.45

# ---------- customers ----------
# acquisition grows over time with a Nov-Dec bump
months = pd.date_range(START, END, freq="MS")
m_w = np.linspace(0.7, 1.6, len(months)) * np.where(months.month.isin([10, 11, 12]), 1.35, 1.0)
m_w = m_w / m_w.sum()
signup_month = rng.choice(months, N_CUSTOMERS, p=m_w)
signup = pd.to_datetime(signup_month) + pd.to_timedelta(rng.integers(0, 28, N_CUSTOMERS), unit="D")
loc_ids = rng.choice(locations.location_id, N_CUSTOMERS, p=locations.weight / locations.weight.sum())
channel = rng.choice(["Organic", "Paid Ads", "Referral", "Social", "Email"], N_CUSTOMERS, p=[0.30, 0.28, 0.14, 0.18, 0.10])
customers = pd.DataFrame(
    {
        "customer_id": [f"C{i:05d}" for i in range(1, N_CUSTOMERS + 1)],
        "signup_date": signup,
        "location_id": loc_ids,
        "acquisition_channel": channel,
        "gender": rng.choice(["F", "M", "Other"], N_CUSTOMERS, p=[0.47, 0.50, 0.03]),
        "age": np.clip(rng.normal(32, 9, N_CUSTOMERS).round(), 18, 70).astype(int),
    }
)
# behaviour: repeat propensity depends on channel; a loyal ~12% buy a lot
chan_rep = {"Organic": 1.0, "Paid Ads": 0.65, "Referral": 1.35, "Social": 0.8, "Email": 1.5}
loyal = rng.random(N_CUSTOMERS) < 0.12
rate = rng.gamma(1.2, 0.18, N_CUSTOMERS) * customers.acquisition_channel.map(chan_rep).values
rate = np.where(loyal, rate * 4 + 0.4, rate)  # expected orders per month

# ---------- orders ----------
orders, items = [], []
oid = 1
cust_loc_region = customers.location_id.map(locations.set_index("location_id").region)
prod_p = (products["pop"] / products["pop"].sum()).values
payment_opts = ["UPI", "Credit Card", "Debit Card", "COD", "Net Banking"]
for i, row in customers.iterrows():
    active_months = max((END - row.signup_date).days / 30.4, 0.5)
    n_orders = 1 + rng.poisson(rate[i] * active_months * 0.55)  # everyone has a first order
    n_orders = min(n_orders, 60)
    dates = [row.signup_date + pd.Timedelta(days=int(rng.integers(0, 3)))]
    for _ in range(n_orders - 1):
        gap = rng.exponential(30.4 / max(rate[i], 0.05) * 1.8)
        d = dates[-1] + pd.Timedelta(days=float(gap) + 1)
        if d > END:
            break
        # seasonality: push some orders into Oct-Dec
        dates.append(d)
    for d in dates:
        if d > END:
            continue
        n_lines = 1 + rng.poisson(0.6)
        chosen = rng.choice(len(products), n_lines, replace=False, p=prod_p)
        # region effect on COD
        reg = cust_loc_region.iloc[i]
        pw = np.array([0.38, 0.17, 0.15, 0.22 if reg in ("East", "North-East", "North") else 0.12, 0.08])
        pay = rng.choice(payment_opts, p=pw / pw.sum())
        status = rng.choice(["Delivered", "Returned", "Cancelled"], p=[0.88, 0.08, 0.04])
        orders.append((f"O{oid:06d}", row.customer_id, d.normalize(), status, pay, round(float(rng.choice([0, 0, 40, 60, 99])), 2)))
        for pi in chosen:
            p = products.iloc[pi]
            qty = int(rng.choice([1, 1, 1, 2, 3], p=[0.4, 0.2, 0.15, 0.17, 0.08]))
            disc = float(rng.choice([0, 0, 0.05, 0.1, 0.2, 0.3], p=[0.3, 0.2, 0.15, 0.15, 0.12, 0.08]))
            items.append((f"O{oid:06d}", p.product_id, qty, p.unit_price, disc))
        oid += 1

orders = pd.DataFrame(orders, columns=["order_id", "customer_id", "order_date", "order_status", "payment_method", "shipping_fee"])
order_items = pd.DataFrame(items, columns=["order_id", "product_id", "quantity", "unit_price", "discount_pct"])

# ---------- inject realistic mess ----------
# 1. duplicate orders & order lines
dup_o = orders.sample(frac=0.015, random_state=1)
orders = pd.concat([orders, dup_o], ignore_index=True).sample(frac=1, random_state=2).reset_index(drop=True)
dup_i = order_items.sample(frac=0.01, random_state=3)
order_items = pd.concat([order_items, dup_i], ignore_index=True)

# 2. missing values
customers.loc[customers.sample(frac=0.04, random_state=4).index, "age"] = np.nan
customers.loc[customers.sample(frac=0.02, random_state=5).index, "location_id"] = np.nan
customers.loc[customers.sample(frac=0.03, random_state=6).index, "gender"] = np.nan
orders.loc[orders.sample(frac=0.015, random_state=7).index, "payment_method"] = np.nan
order_items.loc[order_items.sample(frac=0.01, random_state=8).index, "discount_pct"] = np.nan

# 3. inconsistent categories / casing / whitespace
messy = {"Electronics": ["electronics", "ELECTRONICS", "Electronic"], "Home & Kitchen": ["Home and Kitchen", "home & kitchen"], "Fashion": ["fashion", " Fashion "]}
idx = products.sample(frac=0.25, random_state=9).index
for i in idx:
    c = products.at[i, "category"]
    if c in messy:
        products.at[i, "category"] = rng.choice(messy[c])
orders["payment_method"] = orders["payment_method"].replace({"UPI": "upi"}).where(rng.random(len(orders)) > 0.05, other=orders["payment_method"].str.upper())
orders["order_status"] = orders["order_status"].where(rng.random(len(orders)) > 0.04, other=orders["order_status"].str.lower())
locations.loc[locations.location_id == "L01", "city"] = "bangalore "  # spelling variants of Bengaluru / Mumbai
locations.loc[locations.location_id == "L02", "city"] = "MUMBAI"

# 4. a few bad rows: negative qty, future date
order_items.loc[order_items.sample(12, random_state=11).index, "quantity"] = -1
orders.loc[orders.sample(8, random_state=12).index, "order_date"] = pd.Timestamp("2026-06-01")

for name, df in dict(customers=customers, products=products.drop(columns="pop"), locations=locations, orders=orders, order_items=order_items).items():
    df.to_csv(OUT / f"{name}.csv", index=False)
    print(f"{name:12s} {len(df):>7,d} rows")
