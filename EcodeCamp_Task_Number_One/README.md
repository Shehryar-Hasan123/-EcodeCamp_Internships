### **Table of Contents**
1. [Introduction](#introduction)
2. [Dataset](#dataset)
3. [Installation](#installation)
4. [Project Structure](#project-structure)
5. [Exploratory Data Analysis](#exploratory-data-analysis)
6. [Visualizations](#visualizations)
7. [Conclusion](#conclusion)
8. [Usage](#usage)
9. [Contributing](#contributing)
10. [License](#license)

---

### **1. Introduction**
Customer churn is a significant concern for companies, as losing customers impacts profitability. This project conducts an exploratory data analysis (EDA) on a customer churn dataset to identify patterns and insights that can help reduce churn. The analysis focuses on preprocessing data, exploring key features, and generating visualizations to identify factors contributing to customer churn.

---

### **2. Dataset**
The dataset used in this analysis is the **Telco Customer Churn Dataset**, available from Kaggle. It includes customer details such as demographic information, services subscribed, account information, and whether they churned.

- **Rows**: 7043
- **Columns**: 21
- **Target Variable**: `Churn` (Yes or No)

Key features include:
- **Demographics**: Gender, Senior Citizen, Partner, Dependents
- **Services**: InternetService, StreamingServices
- **Account**: Tenure, Contract, MonthlyCharges, TotalCharges

---

### **3. Installation**
To run the analysis locally, follow these steps:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-repo/customer-churn-eda.git
   cd customer-churn-eda
   ```

2. **Install dependencies**:
   Make sure you have Python and pip installed, then install the necessary libraries:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Jupyter Notebook**:
   You can explore the analysis by running the notebook:
   ```bash
   jupyter notebook
   ```

---

### **4. Project Structure**
The project is structured as follows:

```bash
customer-churn-eda/
│
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
├── notebooks/
│   └── customer_churn_analysis.ipynb
├── visualizations/
│   └── scatter_plot.png
│   └── churn_by_contract.png
│   └── ...
├── README.md
└── requirements.txt
```

- **data/**: Contains the dataset.
- **notebooks/**: Contains the Jupyter Notebook for the EDA.
- **visualizations/**: Stores the generated plots and figures.
- **requirements.txt**: Lists the Python libraries required for the project.

---

### **5. Exploratory Data Analysis**

#### **Data Preprocessing**
We performed the following steps to prepare the data:
1. **Missing Values**: Filled missing values in numerical columns with the median and in categorical columns with the mode.
2. **Outliers**: 
   - Removed outliers in the `MonthlyCharges` and `tenure` columns using the IQR method.
   - Applied log transformation to `MonthlyCharges` to reduce skewness.
3. **Standardization**: Standardized numerical columns using the `StandardScaler`.

#### **Key Findings:**
- Customers with **shorter tenures** and those on **month-to-month contracts** are more likely to churn.
- Higher **MonthlyCharges** correlate with increased churn, but the relationship is less pronounced.

#### **Statistical Summaries**:
| Feature        | Min  | Max  | Mean  | Median | Std   |
|----------------|------|------|-------|--------|-------|
| Tenure (months)| 1    | 72   | 32.37 | 29     | 24.56 |
| MonthlyCharges | 18.25| 118.75| 64.76 | 70.35  | 30.09 |

- **Churn Distribution**: 26.5% of customers have churned, while 73.5% remain.

---

### **6. Visualizations**

The following visualizations help provide insights into the patterns behind churn:

1. **Tenure Distribution**:
   - Shows the distribution of customer tenure, revealing that shorter tenure customers churn more often.



2. **Monthly Charges vs. Tenure**:
   - A scatter plot shows that customers with higher monthly charges may have a higher likelihood of churn, although tenure does not show a strong linear correlation.

  

3. **Correlation Matrix**:
   - Displays the correlation between numerical variables, with `TotalCharges` and `tenure` showing the strongest correlation, as expected.



4. **Monthly Charges by Churn**:
   - A boxplot illustrates that customers who churn tend to have higher monthly charges on average.

  

5. **Churn by Contract Type**:
   - A count plot shows that customers on month-to-month contracts have a significantly higher churn rate.



---

### **7. Conclusion**

The analysis highlights several key factors associated with customer churn:
1. **Contract Type**: Customers with **month-to-month contracts** are far more likely to churn.
2. **Monthly Charges**: Higher monthly charges are linked to higher churn rates.
3. **Tenure**: Customers with shorter tenure are more likely to churn, indicating that retaining customers early is crucial.

#### **Recommendations**:
- Encourage customers to switch from month-to-month contracts to longer-term plans by offering incentives.
- Reevaluate pricing for customers on higher monthly plans to prevent price-sensitive churn.
- Focus on retaining new customers early in their lifecycle.

---

### **8. Usage**

To replicate this analysis:
1. Clone the repository.
2. Install the necessary libraries using `pip install -r requirements.txt`.
3. Open the Jupyter Notebook in `notebooks/customer_churn_analysis.ipynb` to explore the EDA.
4. Run each cell to see the output and visualizations.

---

### **9. Contributing**

We welcome contributions to improve the project. If you wish to contribute:
1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make your changes and commit (`git commit -m "Description of changes"`).
4. Push to your branch (`git push origin feature-branch`).
5. Open a pull request.

---

### **10. License**

This project is licensed under the MIT License. See the `LICENSE` file for more information.

---

This format ensures a clean structure for the report, making it easier to navigate and understand the key elements of the project.
