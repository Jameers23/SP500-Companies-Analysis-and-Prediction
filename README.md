# S&P 500 Companies Analysis with PySpark

This project demonstrates how to process, clean, analyze, and model data using PySpark, focusing on the S&P 500 companies dataset. The project includes data preprocessing, exploratory data analysis (EDA), feature engineering, and machine learning model training.

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Setup](#setup)
- [Data Preprocessing](#data-preprocessing)
- [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
- [Feature Engineering](#feature-engineering)
- [Machine Learning Model](#machine-learning-model)
- [Results](#results)
- [Visualizations](#visualizations)
- [Contributing](#contributing)
- [License](#license)

## Introduction

This project aims to provide a comprehensive understanding of PySpark, from its fundamentals to advanced concepts. The goal is to efficiently process, clean, and analyze a large and complex dataset using PySpark.

## Dataset

The dataset used in this project is the S&P 500 companies dataset, which includes various attributes of the companies such as:
- Exchange
- Symbol
- Shortname
- Longname
- Sector
- Industry
- Current price
- Market cap
- Ebitda
- Revenue growth
- City
- State
- Country
- Full-time employees
- Long business summary
- Weight

## Setup

To run this project, you need to have PySpark installed. Follow these steps to set up the environment:

1. Install PySpark:
    ```sh
    pip install pyspark
    ```

2. Clone the repository:
    ```sh
    git clone https://github.com/yourusername/sp500-analysis-pyspark.git
    cd sp500-analysis-pyspark
    ```

3. Place the `sp500_companies.csv` file in the project directory.

4. Run the main script:
    ```sh
    python main.py
    ```

## Data Preprocessing

### Steps

1. **Setup and Load Data**: Initialize a Spark session and load the dataset.
2. **Handle Missing Values**: Fill missing values using median for numeric columns and mode for categorical columns.
3. **Remove Duplicates**: Ensure no duplicates are present.

### Code Snippet
```python
# Calculate median for numeric columns using approxQuantile
def calculate_median(df, col_name):
    return df.approxQuantile(col_name, [0.5], 0.25)[0]

ebitda_median = calculate_median(df, "Ebitda")
fulltimeemployees_median = calculate_median(df, "Fulltimeemployees")
revenuegrowth_median = calculate_median(df, "Revenuegrowth")
weight_median = calculate_median(df, "Weight")

# Calculate mode for 'State' column
state_mode = df.groupBy("State").count().orderBy("count", ascending=False).first()[0]

# Fill missing values
df = df.fillna({
    'Ebitda': ebitda_median,
    'State': state_mode,
    'Fulltimeemployees': fulltimeemployees_median,
    'Revenuegrowth': revenuegrowth_median,
    'Weight': weight_median
})


## Exploratory Data Analysis (EDA)

### Steps

1. **Distribution of Companies by Sector**: Group by sector and count the number of companies in each sector.
2. **Summary Statistics**: Generate summary statistics for numerical columns.

### Code Snippet
```python
# Group by 'Sector' and count the number of companies in each sector
sector_counts = df.groupBy("Sector").agg(count("Symbol").alias("count")).orderBy("count", ascending=False)
sector_counts.show()

# Describe numerical columns to get summary statistics
df.select("Currentprice", "Marketcap", "Ebitda", "Revenuegrowth", "Fulltimeemployees", "Weight").describe().show()
```

## Feature Engineering

### Steps

1. **Label Encoding**: Convert categorical columns to numeric using `StringIndexer`.
2. **Vector Assembler**: Combine feature columns into a single feature vector.

### Code Snippet
```python
# List of columns to be indexed
categorical_columns = ["Exchange", "Sector", "Industry", "State", "Country"]

for col_name in categorical_columns:
    indexer = StringIndexer(inputCol=col_name, outputCol=f"{col_name}Index")
    df = indexer.fit(df).transform(df)

# Combine feature columns into a single feature vector
feature_columns = ['ExchangeIndex', 'SectorIndex', 'IndustryIndex', 'Marketcap', 'Ebitda', 'Revenuegrowth', 'StateIndex', 'CountryIndex', 'Fulltimeemployees', 'Weight']
assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
df_ml = assembler.transform(df).select("features", col("Currentprice").alias("label"))
```

## Machine Learning Model

### Steps

1. **Train-Test Split**: Split the data into training and test sets.
2. **Train Model**: Train a linear regression model.
3. **Evaluate Model**: Evaluate the model using R^2 and RMSE metrics.

### Code Snippet
```python
# Split the data into training and test sets
train_data, test_data = df_ml.randomSplit([0.8, 0.2])

# Create and train the linear regression model
lr = LinearRegression(featuresCol="features", labelCol="label")
lr_model = lr.fit(train_data)

# Evaluate the model on test data
test_results = lr_model.evaluate(test_data)
print(f"R^2: {test_results.r2}\nRMSE: {test_results.rootMeanSquaredError}")
```

## Results

- **R^2**: Measure of how well the model explains the variability of the target variable.
- **RMSE**: Root Mean Squared Error, indicating the average prediction error.

## Visualizations

### Scatter Plot of Actual vs. Predicted Prices
```python
# Make predictions on the test data
predictions = lr_model.transform(test_data)
pred_df = predictions.toPandas()

# Scatter plot of actual vs. predicted prices
plt.figure(figsize=(10, 6))
plt.scatter(pred_df['label'], pred_df['prediction'])
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Actual vs. Predicted Prices')
plt.show()
```

### Line Plot of Actual vs. Predicted Prices
```python
# Line plot of actual vs. predicted prices
plt.figure(figsize=(10, 6))
n = np.arange(len(pred_df))
plt.plot(n, pred_df['label'], label='Actual', color='blue')
plt.plot(n, pred_df['prediction'], label='Predicted', color='orange')
plt.xlabel('Index')
plt.ylabel('Price')
plt.title('Actual vs. Predicted Prices')
plt.legend()
plt.show()
```

## Contributing

Contributions are welcome! Please submit a pull request or open an issue to discuss any changes.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
```

Feel free to modify the paths, project details, and any other relevant information to match your specific project setup.
