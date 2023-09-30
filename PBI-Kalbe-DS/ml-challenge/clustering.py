import pandas as pd
import numpy as np

# load data
df_customer = pd.read_csv('../dataset/Case Study - Customer.csv', delimiter=';')
df_transaction = pd.read_csv('../dataset/Case Study - Transaction.csv', delimiter=';')

# merge data
merged_df = pd.merge(df_transaction, df_customer, on='CustomerID', how='left')

# convert date to datetime
merged_df['Date'] = pd.to_datetime(merged_df['Date'], format='%d/%m/%Y')

# fill missing values
merged_df['Marital Status'] = merged_df['Marital Status'].fillna(method='ffill')

# convert categorical data to numerical
merged_df['Marital Status'] = merged_df['Marital Status'].apply(lambda x: 1 if x == 'Married' else 0)

# convert income to float
merged_df['Income'] = merged_df['Income'].apply(lambda x: x.replace(',', '.')).astype(float)

# create new dataframe
agg = {
    'TransactionID' : 'count',
    'Qty' : 'sum',
    'TotalAmount' : 'sum'
}

cluster_df = merged_df.groupby('CustomerID').aggregate(agg).reset_index()

# scale data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaled_df = scaler.fit_transform(cluster_df[['TransactionID', 'Qty', 'TotalAmount']])
scaled_df = pd.DataFrame(scaled_df, columns=['TransactionID', 'Qty', 'TotalAmount'])

# build model with kmeans
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

wcss = []
for n in range(1,11):
    model1 = KMeans(
        n_clusters=n, 
        init='k-means++',
        max_iter=100,
        tol=0.0001,
        random_state=100
        )
    model1.fit(scaled_df)
    wcss.append(model1.inertia_)
print(wcss)

kmeans_3 = KMeans(n_clusters=3, init='k-means++', max_iter=300, n_init=10, random_state=100)
kmeans_3.fit(cluster_df)

# add cluster to dataframe
cluster_df['cluster'] = kmeans_3.labels_
cluster_df = cluster_df.drop('CustomerID', axis=1)

cluster_df['CustomerID'] = df_customer['CustomerID']
cluster_df_mean = cluster_df.groupby('cluster').agg({'CustomerID':'count','TransactionID':'mean','Qty':'mean','TotalAmount':'mean'})
cluster_df_mean.sort_values('CustomerID', ascending=False).astype(int)