"""Cleaning + feature pipeline.

clean() reads data/raw/*.csv, fixes the problems, and returns a tidy line-level
DataFrame (one row per order line) plus a cleaning log. Run as a script to write
data/processed/ and data/cleaning_log.csv.
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW, PROC = ROOT / "data" / "raw", ROOT / "data" / "processed"
AS_OF = pd.Timestamp("2025-09-30")

CATEGORY_MAP = {
    "electronics": "Electronics", "electronic": "Electronics",
    "fashion": "Fashion",
    "home & kitchen": "Home & Kitchen", "home and kitchen": "Home & Kitchen",
    "beauty": "Beauty", "books": "Books", "sports": "Sports", "toys": "Toys",
}
CITY_MAP = {"bangalore": "Bengaluru", "mumbai": "Mumbai"}


def clean():
    log = []

    def note(step, before, after, detail=""):
        log.append({"step": step, "rows_before": before, "rows_after": after, "rows_affected": before - after, "detail": detail})

    customers = pd.read_csv(RAW / "customers.csv", parse_dates=["signup_date"])
    products = pd.read_csv(RAW / "products.csv")
    locations = pd.read_csv(RAW / "locations.csv")
    orders = pd.read_csv(RAW / "orders.csv", parse_dates=["order_date"])
    items = pd.read_csv(RAW / "order_items.csv")

    # --- duplicates
    n = len(orders); orders = orders.drop_duplicates("order_id"); note("Drop duplicate orders", n, len(orders), "same order_id repeated")
    n = len(items); items = items.drop_duplicates(); note("Drop duplicate order lines", n, len(items), "exact repeated rows")

    # --- inconsistent categories
    before = products["category"].nunique()
    products["category"] = products["category"].str.strip().str.lower().map(CATEGORY_MAP)
    note("Standardise product categories", before, products["category"].nunique(), f"{before} spellings -> {products['category'].nunique()} categories")
    locations["city"] = locations["city"].str.strip().str.lower().map(lambda c: CITY_MAP.get(c, c.title()))
    orders["order_status"] = orders["order_status"].str.strip().str.title()
    orders["payment_method"] = orders["payment_method"].str.strip().replace({"upi": "UPI", "UPI": "UPI"}).str.title().replace({"Upi": "UPI", "Cod": "COD"})
    note("Standardise status / payment labels", len(orders), len(orders), "case + whitespace fixed")

    # --- invalid rows
    n = len(items); items = items[items["quantity"] > 0]; note("Drop non-positive quantities", n, len(items), "data-entry errors")
    n = len(orders); orders = orders[orders["order_date"] <= AS_OF]; note("Drop future-dated orders", n, len(orders), f"after {AS_OF.date()}")

    # --- missing values
    miss_age = customers["age"].isna().sum()
    customers["age"] = customers["age"].fillna(customers["age"].median())
    customers["gender"] = customers["gender"].fillna("Unknown")
    miss_loc = customers["location_id"].isna().sum()
    customers["location_id"] = customers["location_id"].fillna("UNK")
    orders["payment_method"] = orders["payment_method"].fillna("Unknown")
    miss_disc = items["discount_pct"].isna().sum()
    items["discount_pct"] = items["discount_pct"].fillna(0.0)
    note("Impute missing values", 0, 0, f"age median ({miss_age}), gender/payment 'Unknown', location 'UNK' ({miss_loc}), discount 0 ({miss_disc})")

    # --- orphan lines
    n = len(items); items = items[items["order_id"].isin(orders["order_id"])]; note("Drop lines without a valid order", n, len(items), "orphans after order cleaning")

    # --- merge
    locations = pd.concat([locations, pd.DataFrame([{"location_id": "UNK", "city": "Unknown", "state": "Unknown", "region": "Unknown", "lat": np.nan, "lon": np.nan}])], ignore_index=True)
    df = (
        items.merge(orders, on="order_id")
        .merge(products.drop(columns=["unit_price"]), on="product_id")
        .merge(customers, on="customer_id", how="left")
        .merge(locations.drop(columns=["weight"], errors="ignore"), on="location_id", how="left")
    )

    # --- money
    df["gross"] = df["quantity"] * df["unit_price"]
    df["net"] = df["gross"] * (1 - df["discount_pct"])
    df["cost"] = df["quantity"] * df["unit_cost"]
    # refunds: Returned orders refund the product value, Cancelled orders never complete
    df["is_returned"] = df["order_status"].eq("Returned")
    df["is_cancelled"] = df["order_status"].eq("Cancelled")
    df["revenue"] = np.where(df["order_status"].eq("Delivered"), df["net"], 0.0)
    df["refunded"] = np.where(df["is_returned"], df["net"], 0.0)
    df["profit"] = np.where(df["order_status"].eq("Delivered"), df["net"] - df["cost"], 0.0)
    df["order_month"] = df["order_date"].dt.to_period("M").dt.to_timestamp()
    df["cohort_month"] = df["signup_date"].dt.to_period("M").dt.to_timestamp()
    df["age_group"] = pd.cut(df["age"], [17, 24, 34, 44, 54, 100], labels=["18-24", "25-34", "35-44", "45-54", "55+"])
    note("Final analysis table", len(df), len(df), f"{df['order_id'].nunique():,} orders, {df['customer_id'].nunique():,} customers")
    return df, pd.DataFrame(log)


def rfm(df):
    """RFM segmentation on completed (Delivered) orders."""
    d = df[df["order_status"] == "Delivered"]
    g = d.groupby("customer_id").agg(
        last=("order_date", "max"), frequency=("order_id", "nunique"), monetary=("revenue", "sum")
    )
    g["recency"] = (AS_OF - g["last"]).dt.days
    g["R"] = pd.qcut(g["recency"].rank(method="first"), 4, labels=[4, 3, 2, 1]).astype(int)
    g["F"] = pd.qcut(g["frequency"].rank(method="first"), 4, labels=[1, 2, 3, 4]).astype(int)
    g["M"] = pd.qcut(g["monetary"].rank(method="first"), 4, labels=[1, 2, 3, 4]).astype(int)

    def seg(r):
        if r.R >= 3 and r.F >= 3 and r.M >= 3: return "Champions"
        if r.F >= 3 and r.R >= 2: return "Loyal"
        if r.R >= 3 and r.F <= 2 and r.M >= 3: return "Big Spenders"
        if r.R >= 3: return "New / Promising"
        if r.F >= 3: return "At Risk"
        return "Lost / Hibernating"

    g["segment"] = g.apply(seg, axis=1)
    return g.drop(columns="last").reset_index()


def cohort_retention(df):
    """% of each signup-month cohort that ordered again N months after their signup month."""
    d = df[df["order_status"] == "Delivered"].drop_duplicates(["order_id"])
    d = d.assign(period=((d["order_month"].dt.year - d["cohort_month"].dt.year) * 12 + (d["order_month"].dt.month - d["cohort_month"].dt.month)))
    d = d[d["period"] >= 0]
    sizes = df.drop_duplicates("customer_id").groupby("cohort_month")["customer_id"].nunique()
    act = d.groupby(["cohort_month", "period"])["customer_id"].nunique().unstack(fill_value=0)
    ret = act.div(sizes.reindex(act.index), axis=0) * 100
    # blank out months that have not happened yet (otherwise they show a misleading 0%)
    max_p = (AS_OF.year - ret.index.year) * 12 + (AS_OF.month - ret.index.month)
    for per in ret.columns:
        ret.loc[max_p < per, per] = np.nan
    ret.index = ret.index.strftime("%Y-%m")
    return ret.round(1)


def kpis(df):
    d = df
    delivered = d[d["order_status"] == "Delivered"]
    orders_n = delivered["order_id"].nunique()
    rev = delivered["revenue"].sum()
    per_cust = delivered.groupby("customer_id")["order_id"].nunique()
    return {
        "revenue": rev,
        "orders": orders_n,
        "aov": rev / orders_n if orders_n else 0,
        "customers": delivered["customer_id"].nunique(),
        "repeat_rate": (per_cust > 1).mean() * 100 if len(per_cust) else 0,
        "margin": delivered["profit"].sum() / rev * 100 if rev else 0,
        "return_rate": d.drop_duplicates("order_id")["is_returned"].mean() * 100,
        "refunded": d["refunded"].sum(),
    }


if __name__ == "__main__":
    PROC.mkdir(parents=True, exist_ok=True)
    df, log = clean()
    df.to_csv(PROC / "sales_clean.csv", index=False)
    rfm(df).to_csv(PROC / "rfm.csv", index=False)
    log.to_csv(ROOT / "data" / "cleaning_log.csv", index=False)
    print(log.to_string(index=False))
    k = kpis(df)
    print({a: round(b, 2) for a, b in k.items()})
