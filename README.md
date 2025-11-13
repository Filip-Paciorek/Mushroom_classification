<h1 align="center"><b>Mushroom Classification: Logistic Regression Performance Analysis</b></h1>

<p align="center">
  An analysis of logistic regression models using Ordinal vs. One-Hot encoding to predict mushroom toxicity, with results visualized in an interactive Apache Superset dashboard.
</p>

---

### **Dashboard Demonstration**

<p align="center">
  <video src="dashboard-demo.webm" autoplay loop muted playsinline width="800"></video>
</p>

---

## Table of Contents
- [Project Overview](#project-overview)
- [Tech Stack](#tech-stack)
- [Methodology](#methodology)
  - [Model Differences](#model-differences)
  - [Model Testing](#model-testing)
- [Key Findings & Conclusion](#key-findings--conclusion)
- [How to Reproduce](#how-to-reproduce)

## Project Overview

The goal of this project was to build and evaluate two logistic regression models for predicting whether a mushroom is poisonous or edible based on the UCI Mushroom dataset. The primary focus was to determine the impact of different categorical encoding strategies on model performance, especially under conditions where the most predictive features were restricted.

## Tech Stack

- **Data Analysis:** Python, Pandas, NumPy, Scikit-learn
- **Data Visualization:** Apache Superset
- **Database:** PostgreSQL

## Methodology

### Model Differences

Two primary models were developed to compare encoding effectiveness:

1.  **Model 1 (One-Hot Encoding):** This model uses one-hot encoding for all categorical features, creating a wide dataset with binary columns for each category.
2.  **Model 2 (Ordinal Encoding):** This model uses ordinal encoding, converting categorical features into integer values. This was applied to all features that demonstrated stable predictive power across multiple iterations.

### Model Testing

To rigorously test the models' adaptability, they were evaluated against three distinct feature restriction levels:

-   **Level 1 (Base):** All available features were used. In this scenario, both models achieved high and very similar accuracy.
-   **Level 2 (Mild Restriction):** The most obviously predictive features (e.g., odor) were removed. Here, the One-Hot model's performance degraded more significantly, indicating a harder time adapting, while the Ordinal model remained more robust.
-   **Level 3 (Severe Restriction):** Additional predictive features were removed. At this level, the performance difference between the models diminished. This is likely because the most telling features also had a strong ordinal nature, and removing them left fewer features where the ordinal strategy could provide a distinct advantage.

## Key Findings & Conclusion

The analysis clearly demonstrates that the choice of encoding strategy has a measurable impact on model robustness.

- **Ordinal Encoding provides a performance edge:** While both models perform well with a full feature set, the Ordinal model consistently makes fewer prediction errors when key features are restricted.
- **Context is critical:** In a sensitive domain like predicting mushroom toxicity, minimizing every possible error is paramount. A false positive (classifying a poisonous mushroom as edible) has severe consequences.
- **Recommendation:** The **Ordinal Encoded model (Model 2) is the recommended choice**. It provides a better balance of high accuracy and reliability, proving to be more adaptable in less-than-ideal feature environments.

## How to Reproduce

1.  **Set up Environment:**
    ```bash
    pip install -r requirements.txt
    ```
2.  **Run Modeling:** Execute the analysis notebook to generate the `mushroom_results.csv` file containing the model performance metrics.
3.  **Deploy Dashboard:**
    -   Ensure you have a running instance of Apache Superset.
    -   Upload the `data/processed/mushroom_results.csv` file as a new dataset.
    -   Import the `superset/dashboard_export.zip` file to instantly recreate the dashboard.
