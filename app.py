"""E-commerce Sales & Customer Insights Dashboard (Streamlit)."""
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
import pipeline as pl  # noqa: E402

st.set_page_config(page_title="E-commerce Insights", page_icon="🛒", layout="wide")

# Fixed colour per entity (never repainted by filters)
CAT_COLORS = {
    "Electronics": "#2a78d6", "Fashion": "#eb6834", "Home & Kitchen": "#1baf7a", "Beauty": "#e87ba4",
    "Sports": "#eda100", "Toys": "#4a3aa7", "Books": "#008300",
}
SEG_COLORS = {
    "Champions": "#2a78d6", "Loyal": "#1baf7a", "Big Spenders": "#eda100",
    "New / Promising": "#e87ba4", "At Risk": "#eb6834", "Lost / Hibernating": "#8a8a85",
}
BLUE = "#2a78d6"
INR = lambda v: f"₹{v/1e7:.2f} Cr" if v >= 1e7 else (f"₹{v/1e5:.2f} L" if v >= 1e5 else f"₹{v:,.0f}")


@st.cache_data(show_spinner="Loading data…")
def load():
    f = ROOT / "data" / "processed" / "sales_clean.csv"
    if f.exists():
        df = pd.read_csv(f, parse_dates=["order_date", "signup_date", "order_month", "cohort_month"])
    else:  # first run on a fresh clone
        df, _ = pl.clean()
    log = pd.read_csv(ROOT / "data" / "cleaning_log.csv") if (ROOT / "data" / "cleaning_log.csv").exists() else pl.clean()[1]
    return df, log


df_all, log = load()

# ------------------------------------------------------------------ sidebar filters
st.sidebar.header("Filters")
dmin, dmax = df_all["order_date"].min().date(), df_all["order_date"].max().date()
dr = st.sidebar.date_input("Order date range", (dmin, dmax), min_value=dmin, max_value=dmax)
if len(dr) != 2:
    st.info("Pick an end date to apply the range.")
    st.stop()
regions = st.sidebar.multiselect("Region", sorted(df_all["region"].unique()), default=sorted(df_all["region"].unique()))
cats = st.sidebar.multiselect("Category", sorted(df_all["category"].unique()), default=sorted(df_all["category"].unique()))
chans = st.sidebar.multiselect("Acquisition channel", sorted(df_all["acquisition_channel"].unique()), default=sorted(df_all["acquisition_channel"].unique()))
pays = st.sidebar.multiselect("Payment method", sorted(df_all["payment_method"].unique()), default=sorted(df_all["payment_method"].unique()))
st.sidebar.caption("Data is synthetic (generated with a fixed seed) — see README.")

df = df_all[
    df_all["order_date"].between(pd.Timestamp(dr[0]), pd.Timestamp(dr[1]))
    & df_all["region"].isin(regions) & df_all["category"].isin(cats)
    & df_all["acquisition_channel"].isin(chans) & df_all["payment_method"].isin(pays)
]
if df.empty:
    st.warning("No data for these filters.")
    st.stop()
dl = df[df["order_status"] == "Delivered"]  # completed sales

st.title("🛒 E-commerce Sales & Customer Insights")
st.caption(f"{dr[0]:%d %b %Y} → {dr[1]:%d %b %Y} · {df['order_id'].nunique():,} orders · {df['customer_id'].nunique():,} customers")

# ------------------------------------------------------------------ KPIs
k = pl.kpis(df)
c = st.columns(6)
c[0].metric("Net revenue", INR(k["revenue"]))
c[1].metric("Orders", f"{k['orders']:,}")
c[2].metric("Avg order value", f"₹{k['aov']:,.0f}")
c[3].metric("Repeat-customer rate", f"{k['repeat_rate']:.1f}%")
c[4].metric("Gross margin", f"{k['margin']:.1f}%")
c[5].metric("Return rate", f"{k['return_rate']:.1f}%", help=f"Refunded value: {INR(k['refunded'])}")


def style(fig, h=360):
    fig.update_layout(height=h, margin=dict(l=10, r=10, t=40, b=10), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", legend_title_text="", title_font_size=15)
    fig.update_xaxes(showgrid=False, title=None)
    fig.update_yaxes(gridcolor="rgba(128,128,128,0.2)", title=None)
    return fig


t1, t2, t3, t4, t5, t6 = st.tabs(["📈 Trends", "📦 Products", "🌍 Geography", "👥 Customers", "🔁 Cohorts", "💡 Recommendations"])

# ------------------------------------------------------------------ Trends
with t1:
    m = dl.groupby("order_month").agg(revenue=("revenue", "sum"), orders=("order_id", "nunique")).reset_index()
    m["aov"] = m["revenue"] / m["orders"]
    a, b = st.columns(2)
    f = px.area(m, x="order_month", y="revenue", title="Monthly net revenue (₹)", color_discrete_sequence=[BLUE])
    a.plotly_chart(style(f), width="stretch")
    f = px.line(m, x="order_month", y="orders", title="Monthly orders", markers=True, color_discrete_sequence=[BLUE])
    b.plotly_chart(style(f), width="stretch")
    mc = dl.groupby(["order_month", "category"])["revenue"].sum().reset_index()
    f = px.bar(mc, x="order_month", y="revenue", color="category", title="Revenue by category over time (₹)", color_discrete_map=CAT_COLORS)
    st.plotly_chart(style(f, 400), width="stretch")
    # month-over-month and YoY
    if len(m) > 12:
        yoy = (m["revenue"].iloc[-1] / m["revenue"].iloc[-13] - 1) * 100
        st.caption(f"Latest month vs same month last year: **{yoy:+.1f}%** revenue.")

# ------------------------------------------------------------------ Products
with t2:
    cat = dl.groupby("category").agg(revenue=("revenue", "sum"), profit=("profit", "sum"), units=("quantity", "sum")).reset_index()
    cat["margin_pct"] = cat["profit"] / cat["revenue"] * 100
    a, b = st.columns(2)
    f = px.bar(cat.sort_values("revenue"), x="revenue", y="category", orientation="h", title="Revenue by category (₹)", color="category", color_discrete_map=CAT_COLORS)
    f.update_layout(showlegend=False)
    a.plotly_chart(style(f), width="stretch")
    f = px.scatter(cat, x="revenue", y="margin_pct", size="units", color="category", text="category", title="Revenue vs margin % (bubble = units)", color_discrete_map=CAT_COLORS, size_max=45)
    f.update_traces(textposition="top center")
    f.update_layout(showlegend=False)
    f.update_yaxes(title="Gross margin %")
    f.update_xaxes(title="Revenue (₹)")
    b.plotly_chart(style(f), width="stretch")
    p = dl.groupby(["product_name", "category"]).agg(revenue=("revenue", "sum"), profit=("profit", "sum"), units=("quantity", "sum")).reset_index()
    a, b = st.columns(2)
    top = p.nlargest(10, "revenue")
    f = px.bar(top.iloc[::-1], x="revenue", y="product_name", orientation="h", color="category", title="Top 10 products by revenue", color_discrete_map=CAT_COLORS)
    a.plotly_chart(style(f, 400), width="stretch")
    bottom = p[p["units"] > 0].nsmallest(10, "revenue")
    f = px.bar(bottom.iloc[::-1], x="revenue", y="product_name", orientation="h", color="category", title="Bottom 10 products by revenue", color_discrete_map=CAT_COLORS)
    b.plotly_chart(style(f, 400), width="stretch")
    with st.expander("Category table"):
        st.dataframe(cat.round(1), width="stretch", hide_index=True)

# ------------------------------------------------------------------ Geography
with t3:
    g = dl.groupby(["city", "state", "region", "lat", "lon"], dropna=False).agg(revenue=("revenue", "sum"), customers=("customer_id", "nunique"), orders=("order_id", "nunique")).reset_index()
    g["aov"] = g["revenue"] / g["orders"]
    gm = g.dropna(subset=["lat"])
    f = px.scatter(gm, x="lon", y="lat", size="revenue", color="aov", text="city", hover_name="city",
                   hover_data={"revenue": ":,.0f", "customers": True, "aov": ":,.0f", "lat": False, "lon": False},
                   title="Revenue by city (bubble size) · colour = average order value (₹)", color_continuous_scale="Blues", size_max=55)
    f.update_traces(textposition="top center", marker=dict(line=dict(width=1, color="rgba(128,128,128,0.6)")))
    f.update_xaxes(visible=False)
    f.update_yaxes(visible=False, scaleanchor="x")
    st.plotly_chart(style(f, 480), width="stretch")
    a, b = st.columns(2)
    f = px.bar(g.sort_values("revenue"), x="revenue", y="city", orientation="h", title="Revenue by city (₹)", color_discrete_sequence=[BLUE])
    a.plotly_chart(style(f, 420), width="stretch")
    r = dl.groupby("region").agg(revenue=("revenue", "sum"), customers=("customer_id", "nunique")).reset_index()
    r["rev_per_customer"] = r["revenue"] / r["customers"]
    f = px.bar(r.sort_values("rev_per_customer"), x="rev_per_customer", y="region", orientation="h", title="Revenue per customer by region (₹)", color_discrete_sequence=[BLUE])
    b.plotly_chart(style(f, 420), width="stretch")

# ------------------------------------------------------------------ Customers
rf = pl.rfm(df)
with t4:
    seg = rf.groupby("segment").agg(customers=("customer_id", "count"), revenue=("monetary", "sum"), avg_orders=("frequency", "mean")).reset_index()
    seg["rev_share"] = seg["revenue"] / seg["revenue"].sum() * 100
    a, b = st.columns(2)
    f = px.bar(seg.sort_values("customers"), x="customers", y="segment", orientation="h", color="segment", title="Customers per RFM segment", color_discrete_map=SEG_COLORS)
    f.update_layout(showlegend=False)
    a.plotly_chart(style(f), width="stretch")
    f = px.bar(seg.sort_values("rev_share"), x="rev_share", y="segment", orientation="h", color="segment", title="Share of revenue by segment (%)", color_discrete_map=SEG_COLORS)
    f.update_layout(showlegend=False)
    b.plotly_chart(style(f), width="stretch")
    ch = dl.groupby("acquisition_channel").agg(customers=("customer_id", "nunique"), revenue=("revenue", "sum"))
    multi = dl.groupby(["acquisition_channel", "customer_id"])["order_id"].nunique().gt(1).groupby("acquisition_channel").mean() * 100
    ch["repeat_rate"] = multi
    ch["rev_per_customer"] = ch["revenue"] / ch["customers"]
    ch = ch.reset_index()
    a, b = st.columns(2)
    f = px.bar(ch.sort_values("repeat_rate"), x="repeat_rate", y="acquisition_channel", orientation="h", title="Repeat-customer rate by channel (%)", color_discrete_sequence=[BLUE])
    a.plotly_chart(style(f, 320), width="stretch")
    f = px.bar(ch.sort_values("rev_per_customer"), x="rev_per_customer", y="acquisition_channel", orientation="h", title="Revenue per customer by channel (₹)", color_discrete_sequence=[BLUE])
    b.plotly_chart(style(f, 320), width="stretch")
    st.subheader("Top 20 high-value customers")
    top = rf.nlargest(20, "monetary")[["customer_id", "segment", "frequency", "monetary", "recency"]]
    top.columns = ["Customer", "Segment", "Orders", "Revenue (₹)", "Days since last order"]
    st.dataframe(top.round(0), width="stretch", hide_index=True)
    st.download_button("⬇ Download RFM segments (CSV)", rf.to_csv(index=False), "rfm_segments.csv", "text/csv")

# ------------------------------------------------------------------ Cohorts
with t5:
    ret = pl.cohort_retention(df)
    ret = ret.iloc[:, 1:13]  # drop month 0 (always ~100%); show months 1-12
    f = go.Figure(go.Heatmap(z=ret.values, x=[f"M+{c}" for c in ret.columns], y=ret.index, colorscale="Blues", zmin=0, text=ret.map(lambda v: "" if pd.isna(v) else f"{v:.0f}").values, texttemplate="%{text}", hovertemplate="Cohort %{y}<br>%{x}: %{z:.1f}% active<extra></extra>", colorbar=dict(title="% active")))
    f.update_layout(title="Cohort retention — % of signup-month cohort ordering N months later", yaxis=dict(autorange="reversed"))
    st.plotly_chart(style(f, 640), width="stretch")
    st.caption("Rows = month customers signed up. Columns = months since signup. Month 0 (their first order) is omitted.")

# ------------------------------------------------------------------ Recommendations
with t6:
    st.subheader("Data-backed recommendations (recomputed for your filters)")
    recs = []
    if len(cat) > 1:
        hi_rev = cat.nlargest(1, "revenue").iloc[0]
        if hi_rev["margin_pct"] < cat["margin_pct"].median():
            best = cat.nlargest(1, "margin_pct").iloc[0]
            recs.append(f"**Rebalance towards margin.** {hi_rev['category']} is the #1 category by revenue ({INR(hi_rev['revenue'])}, {hi_rev['revenue']/cat['revenue'].sum()*100:.0f}% of sales) but earns only {hi_rev['margin_pct']:.1f}% margin, while {best['category']} earns {best['margin_pct']:.1f}%. Cross-sell {best['category']} on {hi_rev['category']} order pages and review {hi_rev['category']} discounting.")
        low = cat.nsmallest(1, "revenue").iloc[0]
        recs.append(f"**Fix or prune the weakest category.** {low['category']} brings just {INR(low['revenue'])} ({low['revenue']/cat['revenue'].sum()*100:.1f}% of revenue). Bundle it with high-traffic categories or cut slow SKUs.")
    if len(ch) > 1:
        bc, wc = ch.nlargest(1, "repeat_rate").iloc[0], ch.nsmallest(1, "repeat_rate").iloc[0]
        recs.append(f"**Shift acquisition spend by quality, not volume.** {bc['acquisition_channel']} customers repeat at {bc['repeat_rate']:.0f}% vs {wc['repeat_rate']:.0f}% for {wc['acquisition_channel']}. Move budget towards {bc['acquisition_channel']} and tighten {wc['acquisition_channel']} targeting.")
    champ = seg[seg["segment"] == "Champions"]
    if not champ.empty:
        c0 = champ.iloc[0]
        recs.append(f"**Protect the Champions.** {int(c0['customers']):,} customers ({c0['customers']/seg['customers'].sum()*100:.0f}% of base) drive {c0['rev_share']:.0f}% of revenue. Launch early-access and loyalty perks for them.")
    risk = seg[seg["segment"] == "At Risk"]
    if not risk.empty:
        recs.append(f"**Win back 'At Risk' customers.** {int(risk.iloc[0]['customers']):,} formerly frequent buyers have gone quiet — trigger a time-limited win-back email.")
    if len(r) > 1:
        lo = r[r["region"] != "Unknown"].nsmallest(1, "rev_per_customer").iloc[0]
        hi = r[r["region"] != "Unknown"].nlargest(1, "rev_per_customer").iloc[0]
        recs.append(f"**Close the regional gap.** {hi['region']} yields {INR(hi['rev_per_customer'])} per customer vs {INR(lo['rev_per_customer'])} in {lo['region']}. Test free-shipping thresholds and local campaigns in {lo['region']}.")
    refd = df.groupby("category")["refunded"].sum()
    if refd.sum() > 0:
        w = refd.idxmax()
        recs.append(f"**Cut returns where they cost most.** {k['return_rate']:.1f}% of orders are returned ({INR(k['refunded'])} refunded). {w} alone accounts for {refd.max()/refd.sum()*100:.0f}% of refunded value — improve product pages, specs and QC there first.")
    if not m.empty:
        mo = m.assign(mn=m["order_month"].dt.month).groupby("mn")["revenue"].mean()
        pk = mo.idxmax()
        recs.append(f"**Plan inventory and campaigns around seasonality.** Average revenue peaks in month {pk} ({pd.Timestamp(2000, pk, 1):%B}); build stock and promotions ahead of it.")
    for i, r_ in enumerate(recs, 1):
        st.markdown(f"{i}. {r_}")

    with st.expander("Data-cleaning log"):
        st.dataframe(log, width="stretch", hide_index=True)
