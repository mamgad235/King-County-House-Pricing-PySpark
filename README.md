# 🏡 King County Real Estate Price Prediction

**A Big Data Machine Learning Pipeline built with PySpark.**

This project leverages distributed computing principles to analyze over 21,000 real estate transactions in King County, USA. The objective is to extract market insights and construct a highly accurate predictive model for house prices using Advanced Ensemble Learning.

## 🛠️ Technology Stack
* **Core Framework:** Apache Spark (PySpark)
* **Machine Learning:** PySpark MLlib
* **Data Manipulation:** Pandas
* **Visualization:** Matplotlib, Seaborn

## 📊 Key Market Insights
During the Exploratory Data Analysis (EDA) phase, several critical business insights were uncovered:
1. **The Luxury Market:** Zipcode `98039` is the most premium real estate zone, with an average property value exceeding **$2.16 Million**.
2. **Property Condition Impact:** There is a massive valuation leap based on maintenance. "Condition 1" homes average ~$334k, while "Condition 5" homes command over **$612k**.
3. **Market Capacity:** A robust market exists for large-capacity homes, with 1,935 properties featuring more than 4 bedrooms.

## 🧠 Machine Learning Architecture
To capture the non-linear complexities of real estate pricing, we implemented a **Gradient-Boosted Trees (GBTRegressor)** algorithm within a PySpark Pipeline.

**Pipeline Stages:**
1. **Feature Engineering:** Derived a new `house_age` feature from the build year to provide historical context.
2. **Categorical Encoding:** Utilized `StringIndexer` and `OneHotEncoder` to transform geographic zipcodes into mathematical vectors.
3. **Vector Assembly:** Merged all numerical and encoded features into a single dense vector using `VectorAssembler`.

## 📈 Model Performance
The GBT model achieved excellent predictive accuracy on the hidden 20% test dataset:
* **R-Squared ($R^2$): 0.8016** (Explains >80% of the variance in house prices).
* **RMSE:** **$164,615** (A highly competitive error margin given the multi-million dollar range of the dataset).

---
*Completed as part of the Artificial Intelligence Engineering program.*
