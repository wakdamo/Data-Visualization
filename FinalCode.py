# -*- coding: utf-8 -*-
"""
Created on Wed Aug 12 13:05:39 2020

@author: Amol Wakde
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import plotly.graph_objs as go
import plotly .offline as offline
import plotly.figure_factory as ff

# Importing dataset and examining it
dataset = pd.read_csv("customers.csv")
print(dataset.head())
print(dataset.shape)
print(dataset.info())
print(dataset.describe())

# Plotting Correlation Heatmap
corrs = dataset.corr()
figure = ff.create_annotated_heatmap(
    z=corrs.values,
    x=list(corrs.columns),
    y=list(corrs.index),
    annotation_text=corrs.round(2).values,
    showscale=True)
offline.plot(figure,filename='corrheatmap.html')

### Dropping columns with high correlation + causation
dataset = dataset.drop(['TotalCharges'], axis = 1)
print(dataset.info())
### Converting Categorical features into Numerical features
categorical_features = ['gender','Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup','DeviceProtection','TechSupport','StreamingTV','StreamingMovies','Contract','PaperlessBilling','PaymentMethod']
final_data = pd.get_dummies(dataset, columns = categorical_features)
print(final_data.info())
print(final_data.head(2))

# Divide data into subsets 
# Customer personal information 
subset1 = final_data[['gender_Female','gender_Male',
                      'SeniorCitizen','Partner_No','Partner_Yes',
                      'Dependents_No','Dependents_Yes','tenure','MonthlyCharges']]

#Customers using Internet services are Streaming Tv and Movies online or not
subset2=final_data[['InternetService_DSL','InternetService_Fiber optic','InternetService_No',
                    'gender_Female','gender_Male','StreamingTV_No','StreamingTV_No internet service',
                     'StreamingTV_Yes','StreamingMovies_No','StreamingMovies_No internet service','StreamingMovies_Yes']]

# Support Genderwise analysis of Customer Service signed up
subset3 = final_data[['gender_Female','gender_Male','OnlineSecurity_No',
                      'OnlineSecurity_No internet service','OnlineSecurity_Yes',
                      'OnlineBackup_No','OnlineBackup_No internet service','OnlineBackup_Yes',
                      'DeviceProtection_No','DeviceProtection_No internet service',
                      'DeviceProtection_Yes','TechSupport_No','TechSupport_No internet service',
                      'TechSupport_Yes']]

# Billing and Contract Factor
subset4 = final_data[['Contract_Month-to-month','Contract_One year','Contract_Two year',
                      'PaperlessBilling_No','PaperlessBilling_Yes','PaymentMethod_Bank transfer (automatic)',
                      'PaymentMethod_Credit card (automatic)','PaymentMethod_Electronic check',
                      'PaymentMethod_Mailed check']]


# Normalizing numerical features so that each feature has mean 0 and variance 1
feature_scaler = StandardScaler()
X1 = feature_scaler.fit_transform(subset1)
X2 = feature_scaler.fit_transform(subset2)
X3 = feature_scaler.fit_transform(subset3)
X4 = feature_scaler.fit_transform(subset4)


##Analysis on subset1 - Personal Data
##Finding the number of clusters (K) - Elbow Plot Method
inertia = []
for i in range(1,11):
    kmeans = KMeans(n_clusters = i, random_state = 100)
    kmeans.fit(X1)
    inertia.append(kmeans.inertia_)

plt.plot(range(1, 11), inertia)
plt.title('The Elbow Plot Subset 1')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.show()

# Running KMeans to generate labels
kmeans = KMeans(n_clusters = 2)
kmeans.fit(X1)

# Implementing t-SNE to visualize dataset
tsne = TSNE(n_components = 2, perplexity =50,n_iter=2000)
x_tsne = tsne.fit_transform(X1)

Gender= list(dataset['gender'])
seniorcitizen = list(dataset['SeniorCitizen'])
partner = list(dataset['Partner'])
dependents = list(dataset['Dependents'])
Tenure = list(dataset['tenure'])
monthlycharges = list(dataset['MonthlyCharges'])

data = [go.Scatter(x=x_tsne[:,0], y=x_tsne[:,1], mode='markers',
                    marker = dict(color=kmeans.labels_, colorscale='Rainbow', opacity=0.5),
                               text=[f'gender: {a}; SeniorCitizen :{b}, Partner:{c}, Dependents:{d},Tenure:{e},MonthlyCharges:{f} ' for a,b,c,d,e,f in list(zip(Gender,seniorcitizen,partner,dependents,Tenure,monthlycharges))],
                                 hoverinfo='text')]
layout = go.Layout(title = 't-SNE Dim Reduction:<br>Gender,SeniorCitizen,Partner,Dependets,Tenure,MonthlyCharges</br>', width = 700, height = 600,
                    xaxis = dict(title='First Dimension'),
                    yaxis = dict(title='Second Dimension'))
fig = go.Figure(data=data, layout=layout)
offline.plot(fig,filename='t-SNE1.html')


# Analysis on subset2 - Streaming Tv and Movies online
# Finding the number of clusters (K) - Elbow Plot Method
inertia = []
for i in range(1,11):
    kmeans = KMeans(n_clusters = i, random_state = 100)
    kmeans.fit(X1)
    inertia.append(kmeans.inertia_)

plt.plot(range(1, 11), inertia)
plt.title('The Elbow Plot Subset 2')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.show()

# Running KMeans to generate labels
kmeans = KMeans(n_clusters = 3)
kmeans.fit(X1)

# Implementing t-SNE to visualize dataset
tsne = TSNE(n_components = 3, perplexity =50,n_iter=2000)
x_tsne = tsne.fit_transform(X2)

InternetService = list(dataset['InternetService'])
gender = list(dataset['gender'])
StreamingMovies = list(dataset['StreamingMovies'])
StreamingTV = list(dataset['StreamingTV'])

data = [go.Scatter(x=x_tsne[:,0], y=x_tsne[:,1], mode='markers',
                    marker = dict(color=kmeans.labels_, colorscale='Rainbow', opacity=0.5),
                                text=[f'InternetService: {a}, gender:{b},StreamingMovies :{c}, StreamingTV :{d}' 
                                for a,b,c,d in list(zip(InternetService,gender,
                                StreamingMovies,StreamingTV))],
                                hoverinfo='text')]

layout = go.Layout(title = 't-SNE Plot Dimensionality Reduction: InternetService,gender,StreamingMovies,StreamingTV', width = 800, height = 700,
font=dict(family="Courier New, monospace",size=9, color="RebeccaPurple"),
                    xaxis = dict(title='First Dimension'),
                    yaxis = dict(title='Second Dimension'))
fig = go.Figure(data=data, layout=layout)
offline.plot(fig,filename='t-SNEfinal2.html')



# Analysis on subset3 - Support Service Factor
# Finding the number of clusters (K) - Elbow Plot Method
inertia = []
for i in range(1,11):
    kmeans = KMeans(n_clusters = i, random_state = 100)
    kmeans.fit(X3)
    inertia.append(kmeans.inertia_)

plt.plot(range(1, 11), inertia)
plt.title('The Elbow Plot Subset 3')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.show()

# Running KMeans to generate labels
kmeans = KMeans(n_clusters =3 )
kmeans.fit(X3)

# Implementing t-SNE to visualize dataset
tsne = TSNE(n_components = 3, perplexity =50,n_iter=2000)
x_tsne = tsne.fit_transform(X3)

gender = list(dataset['gender'])
OnlineSecurity = list(dataset['OnlineSecurity'])
OnlineBackup = list(dataset['OnlineBackup'])
DeviceProtection = list(dataset['DeviceProtection'])
TechSupport = list(dataset['TechSupport'])

data = [go.Scatter(x=x_tsne[:,0], y=x_tsne[:,1], mode='markers',
                    marker = dict(color=kmeans.labels_, colorscale='Rainbow', opacity=0.5),text=[f'<b>Gender:</b> {a};<br>Online Security: {b};<br><b>Online Backup:</b> {c};<br><b>Device Protection:</b> {d};<br><b>Technical Support:</b> {e}' for a,b,c,d,e in list(zip( gender,OnlineSecurity,OnlineBackup,DeviceProtection,TechSupport))],
                                  hoverinfo='text')]

layout = go.Layout(title = '<b>t-SNE Plot Dimensionality Reduction: Gender,Online Security,Online Backup,<br>Device Protection and Technical Support</b>', width = 800, height = 700,font=dict(family="Courier New, monospace",size=9, color="RebeccaPurple"),
                    xaxis = dict(title= '<b>First Dimension</b>'),
                    yaxis = dict(title= '<b>Second Dimension</b>'))
fig = go.Figure(data=data, layout=layout)
offline.plot(fig,filename='t-SNEfinal2.html')



# Analysis on subset4 - Support Service Factor
# Finding the number of clusters (K) - Elbow Plot Method
inertia = []
for i in range(1,11):
    kmeans = KMeans(n_clusters = i, random_state = 100)
    kmeans.fit(X4)
    inertia.append(kmeans.inertia_)

plt.plot(range(1, 11), inertia)
plt.title('The Elbow Plot Subset 4')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.show()

# Running KMeans to generate labels
kmeans = KMeans(n_clusters =3  )
kmeans.fit(X4)

# Implementing t-SNE to visualize dataset
tsne = TSNE(n_components = 3, perplexity =30,n_iter=2000)
x_tsne = tsne.fit_transform(X4)

Gender = list(dataset['gender'])
Tenure = list(dataset['tenure'])
PhoneService = list(dataset['PhoneService'])
MultipleLines = list(dataset['MultipleLines'])
Contract = list(dataset['Contract'])
PaperlessBilling = list(dataset['PaperlessBilling'])
PaymentMethod = list(dataset['PaymentMethod'])

data = [go.Scatter(x=x_tsne[:,0], y=x_tsne[:,1], mode='markers',
        marker = dict(color=kmeans.labels_, colorscale='Rainbow', opacity=0.5),
        text=[f' Contract: {a};Tenure:{b};Gender: {c};PhoneService:{d};MultipleLines:{e}; PaperlessBilling:{f}; PaymentMethod:{g}' 
              for a,b,c,d,e,f,g in list(zip(Contract,Tenure,Gender,PhoneService,MultipleLines,PaperlessBilling,PaymentMethod))],
              hoverinfo='text')]

layout = go.Layout(title = '<b>t-SNE Plot Dimensionality Reduction: Gender,Online Security,Online Backup,<br>Device Protection and Technical Support</b>', width = 800, height = 700,font=dict(family="Courier New, monospace",size=9, color="RebeccaPurple"),xaxis = dict(title= '<b>First Dimension</b>'),yaxis = dict(title= '<b>Second Dimension</b>'))

fig = go.Figure(data=data, layout=layout)
offline.plot(fig,filename='t-SNE_cluster0002.html')




