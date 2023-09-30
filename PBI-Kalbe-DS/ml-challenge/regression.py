import pandas as pd
import numpy as np
import warnings 
from statsmodels.tsa.statespace.sarimax import SARIMAX
import pmdarima as pm

# load data
df_customer = pd.read_csv('../dataset/Case Study - Customer.csv', sep=';')
df_product = pd.read_csv('../dataset/Case Study - Product.csv', sep=';')
df_store = pd.read_csv('../dataset/Case Study - Store.csv', sep=';')
df_transaction = pd.read_csv('../dataset/Case Study - Transaction.csv', sep=';')

# merge data
merged_df = pd.merge(df_transaction, df_product, on='ProductID', how='left')
merged_df = pd.merge(merged_df, df_customer, on='CustomerID', how='left')
merged_df = pd.merge(merged_df, df_store, on='StoreID', how='left')

# convert to datetime
merged_df['Date'] = pd.to_datetime(merged_df['Date'], format='%d/%m/%Y')

# fill missing values
merged_df['Marital Status'] = merged_df['Marital Status'].fillna(method='ffill')

# create new dataframe for regression
regression_df = df_transaction.groupby('Date')['Qty'].sum().reset_index()
regression_df['Date'] = pd.to_datetime(regression_df['Date'], format='%d/%m/%Y')
regression_df.sort_values(by='Date', inplace=True)
regression_df.set_index('Date', inplace=True)

# split data into train and test
train = regression_df[:int(0.8*(len(regression_df)))]
test = regression_df[int(0.8*(len(regression_df))):]

# build model auto-fit ARIMA
auto_arima_model = pm.auto_arima(
    train['Qty'],
    seasonal=False,
    stepwise=False,
    suppress_warnings=True,
    trace=True
)

# model fit auto-fit ARIMA
p, d, q = auto_arima_model.order
model_auto = SARIMAX(train['Qty'].values, order=(p,d,q))
model_auto_fit = model_auto.fit(disp=False)

# build model ARIMA manual hyperparameter tuning
from statsmodels.tsa.arima.model import ARIMA
model_manual = ARIMA(train, order=(40, 2, 2))
model_manual_fit = model_manual.fit()

# model predict
model = ARIMA(regression_df, order=(40, 2, 2))
model_fit = model.fit()
forecast = model_fit.forecast(steps=100)
forecast = forecast.round().astype(int)

forecast.describe().astype(int)

# forecast product for next 100 days
warnings.filterwarnings('ignore')

df_product_regression = merged_df[['Qty', 'Date', 'Product Name']]
new = df_product_regression.groupby("Product Name")

df_predict_product = pd.DataFrame({'Date': pd.date_range(start='2023-01-01', periods=100)})

for product_name, group_data in new:
    target_var = group_data['Qty']
    model = ARIMA(target_var, order=(40, 2, 2))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=100)
    forecast = forecast.round().astype(int)
    df_predict_product[product_name] = forecast.values

df_predict_product.set_index('Date', inplace=True)

df_predict_product.describe().astype(int)


