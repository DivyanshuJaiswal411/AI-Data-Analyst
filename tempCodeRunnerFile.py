


import streamlit as st
import pandas as pd
import duckdb
import matplotlib.pyplot as plt

# --- Rule-based NL to SQL ---
def nl_to_sql(question):
    q = question.lower()
    if "total sales by region" in q:
        return "SELECT region, SUM(quantity*price) AS total_sales FROM sales GROUP BY region"
    elif "top products" in q:
        return "SELECT product, SUM(quantity*price) AS revenue FROM sales GROUP BY product ORDER BY revenue DESC LIMIT 5"
    elif "average price" in q:
        return "SELECT AVG(price) AS avg_price FROM sales"
    elif "total orders" in q:
        return "SELECT COUNT(*) AS total_orders FROM sales"
    else:
        return "SELECT * FROM sales LIMIT 5"  # fallback

# --- Dataset Summary ---
def english_summary(df):
    summary = []
    summary.append(f"The dataset has {df.shape[0]} rows and {df.shape[1]} columns.")
    for col in df.columns:
        dtype = df[col].dtype
        if dtype == 'object':
            unique_vals = df[col].nunique()
            summary.append(f"- Column '{col}' is categorical with {unique_vals} unique values.")
        else:
            min_val, max_val = df[col].min(), df[col].max()
            summary.append(f"- Column '{col}' is numeric with values ranging from {min_val} to {max_val}.")
    return "\n".join(summary)

# --- Streamlit App ---
st.title("AI-Powered Data Assistant 🧠📊")

uploaded_file = st.file_uploader("Upload a CSV dataset", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("### Dataset Preview", df.head())

    # Save into DuckDB
    con = duckdb.connect()
    con.execute("CREATE OR REPLACE TABLE sales AS SELECT * FROM df")

    # Dataset Summary
    st.write("### Dataset Summary")
    st.text(english_summary(df))

    # Ask Question
    question = st.text_input("Ask a question about the dataset (e.g., 'total sales by region'):")

    if question:
        sql = nl_to_sql(question)
        st.write("Generated SQL:", sql)

        try:
            result = con.execute(sql).df()
            st.write("### Query Result", result)

            # Plot if possible
            if len(result.columns) == 2:
                result.plot(x=result.columns[0], y=result.columns[1], kind="bar")
                st.pyplot(plt.gcf())
        except Exception as e:
            st.error(f"Error running query: {e}")
