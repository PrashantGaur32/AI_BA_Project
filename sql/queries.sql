
-- Top 5 products by revenue
SELECT product, SUM(revenue) AS total_revenue
FROM sales
GROUP BY product
ORDER BY total_revenue DESC
LIMIT 5;

-- Region-wise sales & profit
SELECT region, SUM(revenue) AS revenue, SUM(profit) AS profit
FROM sales
GROUP BY region
ORDER BY revenue DESC;

-- Customer churn rate by segment
SELECT segment, AVG(churn_flag) AS churn_rate, AVG(clv) AS avg_clv
FROM customers
GROUP BY segment;
