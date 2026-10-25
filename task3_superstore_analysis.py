"""


Dataset : Superstore Sales (10,800 raw rows, 9,994 real orders lines)
Tools   : Python, Pandas, NumPy, SQL (sqlite3), Scikit-learn
Output  : cleaned data, KPI tables, forecast, dashboard_data.json
"""

import json
import sqlite3
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

pd.set_option("display.width", 150)
pd.set_option("display.max_columns", 30)

URL = "https://raw.githubusercontent.com/leonism/sample-superstore/master/data/superstore.csv"

# ------------------------------------------------------------------
# 1. Load data
# ------------------------------------------------------------------
raw = pd.read_csv(URL)
df = raw.copy()
print("Raw shape:", df.shape)
print("Missing values:", int(df.isnull().sum().sum()))
print("Duplicate rows:", df.duplicated().sum())

# ------------------------------------------------------------------
# 2. Clean data
# ------------------------------------------------------------------
# 806 rows have an ID but no order details at all (no date, sales, customer). They are useless,
# so we drop them. This also removes most duplicates, because the empty
# rows were copies of each other.
df = df.dropna(subset=["Order Date", "Sales"]).copy()
df = df.drop_duplicates().reset_index(drop=True)

# Postal code is missing for 11 rows. All of them are Burlington, Vermont,
# so we fill with its real postal code (05401).
df.loc[df["Postal Code"].isnull(), "Postal Code"] = 5401
df["Postal Code"] = df["Postal Code"].astype(int).astype(str).str.zfill(5)

# Dates are text -> convert to datetime. Quantity is a count -> int.
df["Order Date"] = pd.to_datetime(df["Order Date"], format="%m/%d/%Y")
df["Ship Date"] = pd.to_datetime(df["Ship Date"], format="%m/%d/%Y")
df["Quantity"] = df["Quantity"].astype(int)
df["Row ID"] = df["Row ID"].astype(int)

print("\nClean shape:", df.shape)
print("Missing values after cleaning:", int(df.isnull().sum().sum()))
print("Date range:", df["Order Date"].min().date(), "to", df["Order Date"].max().date())

# ------------------------------------------------------------------
# 3. Feature engineering
# ------------------------------------------------------------------
df["Year"] = df["Order Date"].dt.year
df["Month"] = df["Order Date"].dt.month
df["YearMonth"] = df["Order Date"].dt.to_period("M").astype(str)
df["Quarter"] = df["Order Date"].dt.quarter
df["Weekday"] = df["Order Date"].dt.day_name()
df["Ship Days"] = (df["Ship Date"] - df["Order Date"]).dt.days
df["Profit Margin %"] = (df["Profit"] / df["Sales"] * 100).round(2)
df["Is Loss"] = (df["Profit"] < 0).astype(int)
df["Discount Band"] = pd.cut(df["Discount"], bins=[-0.01, 0, 0.2, 0.4, 1],
                             labels=["No discount", "1-20%", "21-40%", "Above 40%"])
df["Unit Price"] = (df["Sales"] / (df["Quantity"] * (1 - df["Discount"]))).round(2)

df.to_csv("superstore_cleaned.csv", index=False)

# ------------------------------------------------------------------
# 4. KPIs
# ------------------------------------------------------------------
total_sales = df["Sales"].sum()
total_profit = df["Profit"].sum()
orders = df["Order ID"].nunique()
customers = df["Customer ID"].nunique()
kpi = {
    "Total Sales": round(total_sales, 2),
    "Total Profit": round(total_profit, 2),
    "Profit Margin %": round(total_profit / total_sales * 100, 2),
    "Total Orders": orders,
    "Total Customers": customers,
    "Average Order Value": round(total_sales / orders, 2),
    "Average Discount %": round(df["Discount"].mean() * 100, 2),
    "Average Ship Days": round(df["Ship Days"].mean(), 2),
    "Loss Making Lines %": round(df["Is Loss"].mean() * 100, 2),
}
print("\n===== KPIs =====")
for k, v in kpi.items():
    print(f"{k:22s}: {v:,}")

def summary(by):
    t = df.groupby(by).agg(Sales=("Sales", "sum"), Profit=("Profit", "sum"),
                           Orders=("Order ID", "nunique"), Qty=("Quantity", "sum"))
    t["Margin %"] = (t["Profit"] / t["Sales"] * 100).round(2)
    return t.round(2)

yearly = summary("Year")
yearly["Sales Growth %"] = (yearly["Sales"].pct_change() * 100).round(2)
region = summary("Region").sort_values("Sales", ascending=False)
category = summary("Category").sort_values("Sales", ascending=False)
subcat = summary("Sub-Category").sort_values("Profit", ascending=False)
segment = summary("Segment")
ship = summary("Ship Mode")
state = summary("State").sort_values("Sales", ascending=False)
discount = summary("Discount Band")
monthly = summary("YearMonth")
month_of_year = summary("Month")
top_products = summary("Product Name").sort_values("Sales", ascending=False).head(10)
top_customers = summary("Customer Name").sort_values("Sales", ascending=False).head(10)
loss_products = summary("Product Name").sort_values("Profit").head(10)

for name, t in [("Yearly", yearly), ("Region", region), ("Category", category),
                ("Sub-category", subcat), ("Segment", segment), ("Ship mode", ship),
                ("Discount band", discount), ("Top 10 products", top_products),
                ("Top 10 customers", top_customers), ("Top 10 loss products", loss_products)]:
    print(f"\n--- {name} ---\n{t}")
print("\n--- Top 5 / Bottom 5 states by profit ---")
print(state.sort_values("Profit").head(5)[["Sales", "Profit", "Margin %"]])
print(state.sort_values("Profit").tail(5)[["Sales", "Profit", "Margin %"]])

# repeat customers
orders_per_cust = df.groupby("Customer ID")["Order ID"].nunique()
print("\nCustomers with 5+ orders: %.1f%%" % ((orders_per_cust >= 5).mean() * 100))
top20 = df.groupby("Customer ID")["Sales"].sum().sort_values(ascending=False)
print("Sales share of top 20%% customers: %.1f%%" % (top20.head(int(len(top20) * .2)).sum() / total_sales * 100))

# ------------------------------------------------------------------
# 5. SQL analysis (SQLite)
# ------------------------------------------------------------------
con = sqlite3.connect(":memory:")
df.assign(**{"Order Date": df["Order Date"].dt.strftime("%Y-%m-%d"),
             "Ship Date": df["Ship Date"].dt.strftime("%Y-%m-%d")}
          ).to_sql("sales", con, index=False)

queries = {
    "Sales and profit by category and region": """
        SELECT Category, Region, ROUND(SUM(Sales),0) AS sales, ROUND(SUM(Profit),0) AS profit
        FROM sales GROUP BY Category, Region ORDER BY profit DESC LIMIT 8""",
    "Sub-categories that lose money": """
        SELECT "Sub-Category", ROUND(SUM(Sales),0) AS sales, ROUND(SUM(Profit),0) AS profit
        FROM sales GROUP BY "Sub-Category" HAVING SUM(Profit) < 0""",
    "Customers with more than 10 orders": """
        SELECT COUNT(*) AS customers FROM (
            SELECT "Customer ID" FROM sales GROUP BY "Customer ID"
            HAVING COUNT(DISTINCT "Order ID") > 10)""",
}
print("\n===== SQL queries =====")
for title, q in queries.items():
    print("\n" + title)
    print(pd.read_sql(q, con))
con.close()

# ------------------------------------------------------------------
# 6. Predictive sales trend (Bonus)
# ------------------------------------------------------------------
# Monthly sales -> linear trend + month-of-year seasonality.
ms = df.groupby("YearMonth")["Sales"].sum().reset_index()
ms["t"] = np.arange(len(ms))
ms["Month"] = pd.PeriodIndex(ms["YearMonth"], freq="M").month

def design(t, months):
    X = pd.get_dummies(pd.Categorical(months, categories=range(1, 13)), dtype=float)
    X.insert(0, "t", np.asarray(t, dtype=float))
    return X.values

# test the model first: train on first 42 months, test on last 6
train, test = ms.iloc[:-6], ms.iloc[-6:]
m = LinearRegression().fit(design(train["t"], train["Month"]), train["Sales"])
pred = m.predict(design(test["t"], test["Month"]))
mape = float(np.mean(np.abs((test["Sales"].values - pred) / test["Sales"].values)) * 100)
print(f"\nForecast check on last 6 months -> MAPE: {mape:.1f}%")

# final model on all 48 months, predict next 6
model = LinearRegression().fit(design(ms["t"], ms["Month"]), ms["Sales"])
future_t = np.arange(len(ms), len(ms) + 6)
future_ym = pd.period_range("2019-01", periods=6, freq="M")
future = model.predict(design(future_t, future_ym.month))
fc = pd.DataFrame({"YearMonth": future_ym.astype(str), "Forecast Sales": future.round(2)})
print(fc)

# ------------------------------------------------------------------
# 7. Save tables + data for the dashboard
# ------------------------------------------------------------------
with pd.ExcelWriter("superstore_kpi_tables.xlsx") as xl:
    pd.Series(kpi, name="Value").to_frame().to_excel(xl, sheet_name="KPIs")
    for name, t in [("Yearly", yearly), ("Monthly", monthly), ("Region", region),
                    ("Category", category), ("SubCategory", subcat), ("Segment", segment),
                    ("ShipMode", ship), ("State", state), ("DiscountBand", discount),
                    ("TopProducts", top_products), ("TopCustomers", top_customers),
                    ("LossProducts", loss_products)]:
        t.to_excel(xl, sheet_name=name)
    fc.to_excel(xl, sheet_name="Forecast", index=False)

def idx(col):
    cats = sorted(df[col].unique())
    return cats, df[col].map({c: i for i, c in enumerate(cats)}).tolist()

months, m_idx = idx("YearMonth")
regions, r_idx = idx("Region")
cats, c_idx = idx("Category")
subs, s_idx = idx("Sub-Category")
segs, g_idx = idx("Segment")
prods, p_idx = idx("Product Name")
custs, u_idx = idx("Customer Name")
ords, o_idx = idx("Order ID")

rows = [[m_idx[i], r_idx[i], c_idx[i], s_idx[i], g_idx[i], p_idx[i], u_idx[i], o_idx[i],
         round(float(df["Sales"].iat[i]), 2), round(float(df["Profit"].iat[i]), 2)]
        for i in range(len(df))]
with open("dashboard_data.json", "w") as f:
    json.dump({"months": months, "regions": regions, "cats": cats, "subs": subs,
               "segs": segs, "prods": prods, "custs": custs,
               "forecast": fc.to_dict("list"), "mape": round(mape, 1),
               "rows": rows}, f, separators=(",", ":"))
print("\nSaved: superstore_cleaned.csv, superstore_kpi_tables.xlsx, dashboard_data.json")
