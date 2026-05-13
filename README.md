# 🏡 King County Real Estate Price Prediction

**A Big Data Machine Learning Pipeline built with PySpark.**

This project leverages distributed computing principles to analyze over 21,000 real estate transactions in King County, USA. The objective is to extract market insights and construct a highly accurate predictive model for house prices using Advanced Ensemble Learning.

## 🛠️ Technology Stack
* **Core Framework:** Apache Spark (PySpark)
* **Machine Learning:** PySpark MLlib (Pipeline, CrossValidator, ParamGridBuilder)
* **Data Manipulation:** Pandas (For driver-safe UI aggregation)
* **Visualization:** Matplotlib, Seaborn

## 📊 Key Market Insights
During our distributed Exploratory Data Analysis (EDA) phase, several critical business insights were uncovered:
1. **Geographic Pricing Hierarchy:** By aggregating the top 10 most expensive zipcodes, we mapped the luxury market, revealing elite zones where average property values exceed **$2.16 Million** (e.g., Zipcode `98039`).
2. **Property Condition Impact:** There is a massive valuation leap based on maintenance. "Condition 1" homes average ~$334k, while "Condition 5" homes command over **$612k**.
3. **Market Capacity:** A robust market exists for large-capacity homes, with 1,935 properties featuring more than 4 bedrooms.

## ⚙️ Machine Learning Architecture
To capture the non-linear complexities of real estate pricing, we designed a progressive 3-stage modeling pipeline.

**Pipeline & Feature Engineering:**
* **Geospatial Mapping:** Explicitly incorporated `lat` and `long` coordinates, allowing the tree-based models to draw complex, invisible boundary boxes around luxury waterfronts and expensive neighborhoods.
* **Feature Refinement:** Derived a `house_age` feature from the build year and dropped redundant columns (like `sqft_above` and `yr_built`) to prevent multicollinearity.
* **Categorical Encoding:** Utilized `StringIndexer` and `OneHotEncoder` to transform over 70 geographic zipcodes into a sparse mathematical matrix.

**Model Progression:**
1. **Linear Regression (Baseline):** Established a strong initial baseline by leveraging the sparse One-Hot Encoded zip code matrix to assign linear dollar-value weights to specific neighborhoods.
2. **Gradient-Boosted Trees (Default - 50 Trees):** Captured complex geographic intersections using 50 sequential decision trees, significantly reducing the overall error margin.
3. **Tuned GBT (Winner):** Utilized PySpark's `CrossValidator` to optimize maximum tree depth (Depth 7) and iteration count (75 trees), forcing the algorithm to find deeper patterns without overfitting.

## 📈 Model Performance
The tuned GBT model achieved elite predictive accuracy on the hidden 20% test dataset, outperforming the statistical baseline by a wide margin:

| Model Architecture | RMSE ($) | R-Squared ($R^2$) |
| :--- | :--- | :--- |
| **1. Linear Regression Baseline** | $167,610 | 0.7943 |
| **2. GBT (Baseline - 50 Trees)** | $137,211 | 0.8614 |
| **3. Tuned GBT (CrossValidation)**| **$134,503** | **0.8674** |

---
*Completed as part of the Artificial Intelligence Engineering program at Mansoura University.* *Under the supervision of Dr. Mohamed Abd Elfattah.*
