# B1. Problem Formulation

## (a) Machine Learning Formulation

The objective is to predict the number of items sold for a given store in a given month under a specific promotional strategy.

The **target variable** is:
- items_sold (continuous numerical variable)

The **candidate input features** include:
- Store characteristics: store_id, store_size, location_type
- Promotion-related variables: promotion_type
- Temporal features: year, month, day_of_week, is_weekend, is_month_end, is_festival
- Market conditions: competition_density

This problem is best formulated as a **supervised regression task,** since the goal is to predict a continuous outcome (number of items sold) based on historical labelled data. Regression is appropriate because the output is not categorical but quantitative, and the business objective is to estimate expected sales volume under different conditions. 

---

## (b) **Machine Learning Formulation**

Using **items sold (sales volume)** as the target variable is more reliable than total revenue because revenue can be influenced by pricing strategies, discounts, and product mix. For example, a heavy discount may increase units sold but reduce revenue per item, leading to misleading conclusions about promotion effectiveness.

In contrast, items sold directly captures **customer response and demand,** making it a cleaner measure of how effective a promotion is in driving purchases.

This illustrates a broader principle in machine learning:
    **The target variable should directly reflect the underlying objective of the problem and       should not be distorted by external or confounding factors.**

Choosing the right target ensures that the model learns meaningful patterns rather than noise introduced by unrelated variables.

---

## (c) **Alternate Modelling Strategy**

Instead of using a single global model across all 50 stores, a more effective approach is a **segmented or hierarchical modelling strategy.**

One approach is to build:

- Separate models for different **store segments** (e.g., urban, semi-urban, rural), or
- A single model with strong **interaction effects** between location_type, store_size, and promotion_type

This is important because customer behaviour, purchasing power, and responsiveness to promotions vary significantly across locations. A global model may average out these differences and fail to capture local patterns.

By incorporating segmentation or interactions, the model can better reflect **heterogeneity across stores,** leading to more accurate and actionable predictions.

---
---

# B2. Data and EDA Strategy

## (a) Data Integration and Dataset Design

The raw data is distributed across four tables: transactions, store attributes, promotion details, and calendar data. These would be combined using appropriate keys to create a unified modelling dataset.

- The **transactions table** would serve as the base, as it contains the target variable (`items_sold`)
- The **store attributes table** would be joined using `store_id`
- The **promotion details table** would be joined using `promotion_type`
- The **calendar table** would be joined using `transaction_date`

The final dataset would be structured at the level of:
> **One row per store per day (or per transaction date)**

This ensures that each observation captures the interaction between store characteristics, promotion type, and time-specific factors.

Before modelling, the following aggregations may be performed:
- Aggregate daily transactions to monthly level (if required by business context)
- Compute average or total `items_sold` per store per promotion
- Derive rolling averages (e.g., past 7-day or 30-day sales trends)
- Create lag features (previous period sales)

These steps help capture temporal patterns and stabilise noise in the data.

---

## (b) Exploratory Data Analysis (EDA)

Before building the model, the following analyses would be conducted:

1. **Target Variable Distribution (Histogram of `items_sold`)**  
   This helps identify skewness, outliers, or unusual patterns in sales volume.  
   If the distribution is highly skewed, transformations (e.g., log) may be considered.

2. **Promotion Type vs Sales (Boxplot or Bar Chart)**  
   This reveals how different promotions impact sales.  
   Insights here can guide feature importance expectations and interaction features.

3. **Time-based Trends (Line Plot over Time)**  
   Plotting sales across time helps identify seasonality, trends, and festival effects.  
   This informs the importance of temporal features like `month`, `is_festival`, etc.

4. **Store Segmentation Analysis (Sales by Location Type / Store Size)**  
   This shows how different store categories perform.  
   It supports the need for interaction terms or segmented modelling.

5. **Correlation Heatmap (Numerical Features)**  
   Helps detect multicollinearity and relationships between variables.  
   Highly correlated features may need to be handled carefully.

Each of these analyses directly informs feature engineering decisions and model design.

---

## (c) Handling Promotion Imbalance

If 80% of transactions occur without promotions, the dataset is imbalanced with respect to the `promotion_type` variable.

This imbalance may cause the model to:
- Bias towards predicting outcomes similar to non-promotion scenarios
- Underestimate the true impact of promotional strategies

To address this, the following steps can be taken:

- Ensure proper representation using **stratified sampling (if applicable)**
- Use **feature engineering** to highlight promotion effects (interaction terms)
- Evaluate model performance separately for promotional vs non-promotional cases
- Consider **reweighting or resampling techniques** if necessary

This ensures that the model learns meaningful patterns for both common and rare scenarios.

---
---

# B3. Model Evaluation and Deployment

## (a) Train-Test Split and Evaluation Metrics

Given that the data is time-based (monthly store-level data over three years), a **temporal train-test split** should be used instead of a random split.

A suitable approach would be:
- Use the first ~80% of the timeline (earlier months) as the training set
- Use the most recent ~20% as the test set

A random split is inappropriate because it would mix past and future data, leading to **data leakage**. The model would indirectly learn from future information, resulting in overly optimistic performance that would not generalise to real-world predictions.

The following evaluation metrics should be used:

- **RMSE (Root Mean Squared Error)**  
  Measures the average magnitude of error, with higher penalty for large errors.  
  In this context, it reflects how far predicted sales deviate from actual sales, especially highlighting large forecasting mistakes.

- **MAE (Mean Absolute Error)**  
  Measures the average absolute difference between predicted and actual values.  
  It is more interpretable and robust to outliers, providing a clear estimate of average prediction error.

In a business context:
- Lower RMSE indicates fewer large prediction errors, which is important for avoiding poor promotion decisions
- Lower MAE indicates consistent prediction accuracy across stores and months

---

## (b) Explaining Model Recommendations Using Feature Importance

To understand why the model recommends different promotions for the same store in different months, feature importance and contextual analysis should be used.

The investigation would involve:
- Examining the **feature importance scores** from the Random Forest model to identify key drivers (e.g., `month`, `is_festival`, `promotion_type`, `competition_density`)
- Comparing feature values for Store 12 in March vs December

For example:
- December may show higher values for `is_festival` or seasonal demand, making loyalty-based promotions more effective
- March may reflect lower demand or different customer behaviour, making direct discounts more impactful

To communicate this to the marketing team:
- Translate technical insights into business language (e.g., “December has higher festive demand, so loyalty incentives perform better”)
- Use simple visuals or comparisons to show how conditions differ across months
- Emphasise that the model adapts recommendations based on **changing context**, not randomness

This builds trust in the model’s decisions.

---

## (c) Deployment and Monitoring Strategy

To deploy the model in production without retraining each month, the following pipeline should be implemented:

### Model Saving
- Save the trained pipeline (including preprocessing and model) using tools like `joblib` or `pickle`
- This ensures consistency between training and inference

### Data Preparation
- At the start of each month, collect new data for all 50 stores
- Apply the same preprocessing steps (feature engineering + scaling) using the saved pipeline
- Generate predictions for `items_sold` under different promotion scenarios

### Recommendation Generation
- For each store, select the promotion that yields the highest predicted sales
- Output recommendations in a usable format (dashboard or report)

### Monitoring and Maintenance
- Track model performance over time using metrics like RMSE and MAE on new data
- Monitor for **data drift** (changes in input feature distributions)
- Monitor for **concept drift** (changes in relationship between features and target)

### Retraining Strategy
- Retrain the model periodically (e.g., every few months) or when performance drops beyond a threshold
- Incorporate new data to keep the model up-to-date

This ensures that the model remains accurate, reliable, and aligned with evolving business conditions.
