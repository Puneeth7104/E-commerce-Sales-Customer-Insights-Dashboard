"""Build the static report (docs/index.html) for GitHub Pages and a PDF copy.

Usage:  python src/build_report.py [--app-url https://your-app.streamlit.app] [--pdf]
"""
import argparse
import shutil
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.offline

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pipeline as pl  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"
CAT = {"Electronics": "#2a78d6", "Fashion": "#eb6834", "Home & Kitchen": "#1baf7a", "Beauty": "#e87ba4", "Sports": "#eda100", "Toys": "#4a3aa7", "Books": "#008300"}
SEG = {"Champions": "#2a78d6", "Loyal": "#1baf7a", "Big Spenders": "#eda100", "New / Promising": "#e87ba4", "At Risk": "#eb6834", "Lost / Hibernating": "#8a8a85"}
BLUE = "#2a78d6"
inr = lambda v: f"₹{v/1e7:.2f} Cr" if v >= 1e7 else f"₹{v/1e5:.2f} L"

ap = argparse.ArgumentParser()
ap.add_argument("--app-url", default="")
ap.add_argument("--pdf", action="store_true")
args = ap.parse_args()

df, log = pl.clean()
dl = df[df["order_status"] == "Delivered"]
k = pl.kpis(df)
rf = pl.rfm(df)


def sty(f, h=360):
    f.update_layout(height=h, margin=dict(l=10, r=10, t=50, b=10), paper_bgcolor="white", plot_bgcolor="white", font=dict(family="Inter, Segoe UI, sans-serif", color="#0b0b0b"), title_font_size=15, legend_title_text="")
    f.update_xaxes(showgrid=False, title=None)
    f.update_yaxes(gridcolor="#eeeeec", title=None)
    return f


charts = {}
m = dl.groupby("order_month").agg(revenue=("revenue", "sum")).reset_index()
charts["trend"] = sty(px.area(m, x="order_month", y="revenue", title="Monthly net revenue (₹)", color_discrete_sequence=[BLUE]))
cat = dl.groupby("category").agg(revenue=("revenue", "sum"), profit=("profit", "sum"), units=("quantity", "sum")).reset_index()
cat["margin_pct"] = cat["profit"] / cat["revenue"] * 100
f = px.scatter(cat, x="revenue", y="margin_pct", size="units", color="category", text="category", color_discrete_map=CAT, size_max=50, title="Category: revenue (₹) vs gross margin % — bubble = units")
f.update_traces(textposition="top center"); f.update_layout(showlegend=False)
charts["catmargin"] = sty(f, 400)
p = dl.groupby(["product_name", "category"])["revenue"].sum().reset_index().nlargest(10, "revenue")
charts["top"] = sty(px.bar(p.iloc[::-1], x="revenue", y="product_name", orientation="h", color="category", color_discrete_map=CAT, title="Top 10 products by revenue (₹)"), 400)
g = dl.groupby("city")["revenue"].sum().reset_index().query("city != 'Unknown'")
charts["city"] = sty(px.bar(g.sort_values("revenue"), x="revenue", y="city", orientation="h", color_discrete_sequence=[BLUE], title="Revenue by city (₹)"), 420)
seg = rf.groupby("segment").agg(customers=("customer_id", "count"), revenue=("monetary", "sum")).reset_index()
seg["rev_share"] = seg["revenue"] / seg["revenue"].sum() * 100
seg["cust_share"] = seg["customers"] / seg["customers"].sum() * 100
s2 = seg.melt("segment", ["cust_share", "rev_share"], var_name="metric", value_name="pct").replace({"cust_share": "% of customers", "rev_share": "% of revenue"})
f = px.bar(s2, x="pct", y="segment", color="metric", barmode="group", orientation="h", title="RFM segments: share of customers vs share of revenue (%)", color_discrete_sequence=[BLUE, "#eb6834"])
charts["rfm"] = sty(f, 400)
ret = pl.cohort_retention(df).iloc[:, 1:13]
f = go.Figure(go.Heatmap(z=ret.values, x=[f"M+{c}" for c in ret.columns], y=ret.index, colorscale="Blues", zmin=0, colorbar=dict(title="% active"), hovertemplate="Cohort %{y}<br>%{x}: %{z:.1f}%<extra></extra>"))
f.update_layout(title="Cohort retention — % of signup-month cohort ordering N months later", yaxis=dict(autorange="reversed"))
charts["cohort"] = sty(f, 560)
ch = dl.groupby("acquisition_channel").agg(customers=("customer_id", "nunique"), revenue=("revenue", "sum"))
ch["repeat_rate"] = dl.groupby(["acquisition_channel", "customer_id"])["order_id"].nunique().gt(1).groupby("acquisition_channel").mean() * 100
ch = ch.reset_index()
charts["channel"] = sty(px.bar(ch.sort_values("repeat_rate"), x="repeat_rate", y="acquisition_channel", orientation="h", color_discrete_sequence=[BLUE], title="Repeat-customer rate by acquisition channel (%)"), 320)

DOCS.mkdir(exist_ok=True)
shutil.copy(Path(plotly.offline.__file__).parent.parent / "package_data" / "plotly.min.js", DOCS / "plotly.min.js")
div = {n: f.to_html(full_html=False, include_plotlyjs=False, config={"displayModeBar": False, "responsive": True}) for n, f in charts.items()}

best_m, worst_m = cat.nlargest(1, "margin_pct").iloc[0], cat.nsmallest(1, "margin_pct").iloc[0]
top_rev = cat.nlargest(1, "revenue").iloc[0]
low_rev = cat.nsmallest(1, "revenue").iloc[0]
champ = seg[seg.segment == "Champions"].iloc[0]
bc, wc = ch.nlargest(1, "repeat_rate").iloc[0], ch.nsmallest(1, "repeat_rate").iloc[0]
refd = df.groupby("category")["refunded"].sum()
rg = dl.groupby("region").agg(r=("revenue", "sum"), c=("customer_id", "nunique")).query("index != 'Unknown'")
rg["rpc"] = rg.r / rg.c
rev_south_west = rg.loc[["South", "West"], "r"].sum() / rg.r.sum() * 100
risk = seg[seg.segment == "At Risk"].iloc[0]
recs = [
    f"<b>Rebalance towards margin.</b> {top_rev.category} is the biggest category ({inr(top_rev.revenue)}, {top_rev.revenue/cat.revenue.sum()*100:.0f}% of sales) but earns only {top_rev.margin_pct:.1f}% margin vs {best_m.margin_pct:.1f}% for {best_m.category}. Cross-sell {best_m.category} to {top_rev.category} buyers and review {top_rev.category} discounting.",
    f"<b>Fix or prune {low_rev.category}.</b> It brings only {inr(low_rev.revenue)} ({low_rev.revenue/cat.revenue.sum()*100:.1f}% of revenue) at {low_rev.margin_pct:.0f}% margin — bundle it or cut slow SKUs.",
    f"<b>Buy quality, not volume, in acquisition.</b> {bc.acquisition_channel} customers repeat at {bc.repeat_rate:.0f}% vs {wc.repeat_rate:.0f}% for {wc.acquisition_channel}. Shift budget towards {bc.acquisition_channel} and tighten {wc.acquisition_channel} targeting.",
    f"<b>Protect Champions.</b> {int(champ.customers):,} customers ({champ.cust_share:.0f}% of base) generate {champ.rev_share:.0f}% of revenue — add loyalty tiers and early access.",
    f"<b>Win back At-Risk buyers.</b> {int(risk.customers):,} formerly frequent customers have gone quiet; a time-limited win-back email is the cheapest revenue available.",
    f"<b>Grow beyond the core.</b> South and West deliver {rev_south_west:.0f}% of revenue. East and North-East are under-penetrated — test local campaigns and free-shipping thresholds.",
    f"<b>Attack returns where refunds concentrate.</b> {k['return_rate']:.1f}% of orders are returned ({inr(k['refunded'])} refunded); {refd.idxmax()} is {refd.max()/refd.sum()*100:.0f}% of refunded value.",
]
kp = [("Net revenue", inr(k["revenue"])), ("Orders", f"{k['orders']:,}"), ("Avg order value", f"₹{k['aov']:,.0f}"), ("Repeat-customer rate", f"{k['repeat_rate']:.1f}%"), ("Gross margin", f"{k['margin']:.1f}%"), ("Return rate", f"{k['return_rate']:.1f}%")]
kp_html = "".join(f'<div class="kpi"><span>{a}</span><b>{b}</b></div>' for a, b in kp)
rec_html = "".join(f"<li>{r}</li>" for r in recs)
log_html = log[["step", "rows_affected", "detail"]].to_html(index=False, border=0, classes="log")
app_btn = f'<a class="btn" href="{args.app_url}">Open the interactive dashboard →</a>' if args.app_url else '<span class="note">Interactive dashboard link: add with <code>--app-url</code> after deploying to Streamlit Cloud.</span>'

html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>E-commerce Sales &amp; Customer Insights</title>
<script src="plotly.min.js"></script>
<style>
:root{{--bg:#fcfcfb;--ink:#0b0b0b;--mute:#52514e;--line:#e6e5e0;--accent:#2a78d6}}
@media (prefers-color-scheme:dark){{:root{{--bg:#1a1a19;--ink:#fff;--mute:#c3c2b7;--line:#383835}}}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:16px/1.55 Inter,"Segoe UI",system-ui,sans-serif}}
main{{max-width:1080px;margin:0 auto;padding:32px 16px 64px}}h1{{font-size:clamp(26px,4vw,38px);margin:0 0 4px}}h2{{margin:44px 0 8px;font-size:22px}}
.sub{{color:var(--mute);margin:0 0 20px}}.kpis{{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin:20px 0}}
.kpi{{border:1px solid var(--line);border-radius:10px;padding:14px}}.kpi span{{display:block;color:var(--mute);font-size:13px}}.kpi b{{font-size:26px}}
.grid{{display:grid;grid-template-columns:1fr 1fr;gap:16px}}@media(max-width:800px){{.grid{{grid-template-columns:1fr}}}}
.card{{background:#fff;border:1px solid var(--line);border-radius:10px;padding:6px;overflow:hidden}}
.btn{{display:inline-block;background:var(--accent);color:#fff;padding:10px 18px;border-radius:8px;text-decoration:none;font-weight:600}}
.note{{color:var(--mute);font-size:14px}}li{{margin:10px 0}}table.log{{border-collapse:collapse;width:100%;font-size:14px}}table.log td,table.log th{{padding:6px 8px;border-bottom:1px solid var(--line);text-align:left}}
.warn{{border-left:4px solid #eda100;padding:8px 14px;background:rgba(237,161,0,.08);border-radius:0 8px 8px 0;font-size:14px}}
@media print{{.btn,.note{{display:none}}.card{{break-inside:avoid}}h2{{break-after:avoid}}}}
</style></head><body><main>
<h1>E-commerce Sales &amp; Customer Insights</h1>
<p class="sub">Jan 2023 – Sep 2025 · {df.order_id.nunique():,} orders · {df.customer_id.nunique():,} customers · 98 products · 14 cities</p>
<p>{app_btn}</p>
<div class="warn"><b>About the data:</b> no real dataset was provided, so this analysis uses a synthetic dataset generated by <code>src/generate_data.py</code> (fixed seed, deliberately messy). Figures illustrate the method, not a real business.</div>
<div class="kpis">{kp_html}</div>
<h2>1. Revenue trend</h2><div class="card">{div['trend']}</div>
<h2>2. Products &amp; categories</h2><div class="grid"><div class="card">{div['catmargin']}</div><div class="card">{div['top']}</div></div>
<h2>3. Geography</h2><div class="card">{div['city']}</div>
<h2>4. Customer segments</h2><div class="grid"><div class="card">{div['rfm']}</div><div class="card">{div['channel']}</div></div>
<h2>5. Cohort retention</h2><div class="card">{div['cohort']}</div>
<h2>6. Recommendations</h2><ol>{rec_html}</ol>
<h2>Method</h2>
<p>Order, customer, product and location tables were merged into one order-line table. Cleaning steps are logged below. Revenue counts <b>Delivered</b> orders only; <b>Returned</b> orders are treated as refunds and <b>Cancelled</b> orders never complete. Net revenue = quantity × price × (1 − discount); margin uses unit cost. Repeat-customer rate = customers with 2+ delivered orders. Customers are segmented with RFM quartiles (recency, frequency, monetary).</p>
{log_html}
</main></body></html>"""
(DOCS / "index.html").write_text(html, encoding="utf-8")
(DOCS / ".nojekyll").write_text("")
print("wrote docs/index.html")

if args.pdf:
    from playwright.sync_api import sync_playwright
    with sync_playwright() as pw:
        b = pw.chromium.launch()
        pg = b.new_page(viewport={"width": 740, "height": 900})
        pg.emulate_media(media="print"); pg.goto((DOCS / "index.html").as_uri())
        pg.wait_for_timeout(2500)
        pg.pdf(path=str(DOCS / "report.pdf"), format="A4", print_background=True, margin=dict(top="12mm", bottom="12mm", left="10mm", right="10mm"))
        b.close()
    print("wrote docs/report.pdf")
