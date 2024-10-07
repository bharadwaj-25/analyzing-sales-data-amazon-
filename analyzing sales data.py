#!/usr/bin/env python
# coding: utf-8

# # LOAD DATA SET

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# # loading data

# In[2]:


df = pd.read_csv(r"C:\Users\lalit\Downloads\amazon.csv")


# In[3]:


print(df.head(2))


# In[4]:


print(df.info())


# # data cleaning

# In[5]:


missing_values = df.isnull().sum()
print("Missing Values:\n", missing_values)


# In[6]:


df['rating_count'] = df['rating_count'].str.replace(',', '').astype(float)

print(df['rating_count'].head())


# In[7]:


df['rating_count'] = pd.to_numeric(df['rating_count'], errors='coerce')

print(df['rating_count'].head())


# In[8]:


df['rating_count'].fillna(df['rating_count'].mean(), inplace=True)


# In[9]:


df.dropna(subset=['product_id', 'user_id'], inplace=True)


# # discount distribution

# In[10]:


df['actual_price'] = df['actual_price'].replace('[^0-9.]', '', regex=True).astype(float)
df['discounted_price'] = df['discounted_price'].replace('[^0-9.]', '', regex=True).astype(float)

df['discount_amount'] = df['actual_price'] - df['discounted_price']

print(df[['actual_price', 'discounted_price', 'discount_amount']].head())


# In[11]:


plt.figure(figsize=(10, 6))
sns.histplot(df['discount_amount'], bins=30, kde=True)
plt.title('Distribution of Discount Amount')
plt.xlabel('Discount Amount')
plt.ylabel('Frequency')
plt.show()


# In[12]:


df['discount_percentage'] = df['discount_percentage'].str.replace('%', '').astype(float)

avg_discount_by_category = df.groupby('category')['discount_percentage'].mean().reset_index()

print("Average Discount by Category:\n", avg_discount_by_category)


# In[13]:


print("Average Discount by Category:\n", avg_discount_by_category)


# In[14]:


top_products_by_reviews = df.groupby('product_name')['rating_count'].sum().sort_values(ascending=False).head(10)
print("Top 10 Products by Review Count:\n", top_products_by_reviews)


# # top products by reviews

# In[38]:


plt.figure(figsize=(16, 4))  
sns.barplot(x=top_products_by_reviews.index, y=top_products_by_reviews.values)
plt.title('Top 10 Products by Review Count')
plt.xlabel('Product Name')
plt.ylabel('Review Count')


plt.xticks(rotation=45, ha='right')

plt.tight_layout()

plt.show()


# # ratings distribution of products

# In[17]:


df['rating'] = pd.to_numeric(df['rating'], errors='coerce')

print(df['rating'].isna().sum())


# In[18]:


avg_rating_by_category = df.groupby('category')['rating'].mean().reset_index()
print("Average Rating by Category:\n", avg_rating_by_category)


# In[23]:


top_20_avg_rating_by_category = avg_rating_by_category.sort_values(by='rating', ascending=False).head(20)

plt.figure(figsize=(14, 7))  
sns.barplot(x='category', y='rating', data=top_20_avg_rating_by_category)
plt.title('Top 20 Categories by Average Rating')
plt.xlabel('Category')
plt.ylabel('Average Rating')

plt.ylim(top_20_avg_rating_by_category['rating'].min() - 0.1, top_20_avg_rating_by_category['rating'].max() + 0.1)

plt.xticks(rotation=45, ha='right')  


plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.2f}'))

plt.tight_layout()  
plt.show()


# # correlation matrix

# In[22]:


correlation_matrix = df[['discount_percentage', 'rating_count', 'rating']].corr()
print("Correlation Matrix:\n", correlation_matrix)

plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix for Sales Data')
plt.show()



# In[ ]:




