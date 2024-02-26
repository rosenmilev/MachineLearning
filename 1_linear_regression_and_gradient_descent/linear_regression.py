from urllib.request import urlretrieve
import pandas as pd
import plotly.express as px
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.linear_model import LinearRegression, SGDRegressor

# medical_charges_url = 'https://raw.githubusercontent.com/JovianML/opendatasets/master/data/medical-charges.csv'
#
#
# urlretrieve(medical_charges_url, 'medical.csv')


def load_plot(fig):
	load_plots = False
	if load_plots:
		fig.show()




# Create pandas data frame
medical_df = pd.read_csv('medical.csv')

# Examine data quality
# print(medical_df.describe())

# Some style settings of plotting libraries

sns.set_style('darkgrid')
matplotlib.rcParams['font.size'] = 14
matplotlib.rcParams['figure.figsize'] = (10, 6)
matplotlib.rcParams['figure.facecolor'] = '#00000000'

# 1. Examine and analyze data
# Age column

print(medical_df.age.describe())

# Show histogram of age

fig = px.histogram(medical_df,
				   x='age',
				   marginal='box',
				   nbins=47,
				   title='Distribution of Age')
fig.update_layout(bargap=0.1)
load_plot(fig)

# BMI column

fig = px.histogram(medical_df,
				   x='bmi',
				   marginal='box',
				   color_discrete_sequence=['red'],
				   title='Distribution of BMI ')
fig.update_layout(bargap=0.1)
load_plot(fig)

# Charges column
# Charges by Smoker/Non-smoker
fig = px.histogram(medical_df,
				   x='charges',
				   marginal='box',
				   color='smoker',
				   color_discrete_sequence=['green', 'grey'],
				   title='Annual Medical Charges By Smokers/Non-Smokers'
				   )

fig.update_layout(bargap=0.1)
load_plot(fig)

# Charges by sex

fig = px.histogram(medical_df,
				   x='charges',
				   marginal='box',
				   color='sex',
				   color_discrete_sequence=['blue', 'red'],
				   title='Annual Medical Charges By Sex')
fig.update_layout(bargap=0.1)
load_plot(fig)

# Smoker

fig = px.histogram(medical_df, x='smoker', color='sex', title='Smoker')
load_plot(fig)

# Sex

fig = px.histogram(medical_df, x='charges', color='children', title='Sex', nbins=20)
fig.update_layout(bargap=0.1)
load_plot(fig)

# Age and Charges scatter

fig = px.scatter(medical_df,
				 x='age',
				 y='charges',
				 color='smoker',
				 opacity=0.8,
				 hover_data=['sex'],
				 title='Age vs. Charges')
fig.update_traces(marker_size=5)
load_plot(fig)

# Charges and BMI scatter

fig = px.scatter(medical_df,
				 x='bmi',
				 y='charges',
				 color='smoker',
				 opacity=0.8,
				 hover_data=['sex'],
				 title='BMI vs Charges')
fig.update_traces(marker_size=5)
load_plot(fig)

# Examine relationship between charges and children
plt.switch_backend('agg')
fig = px.violin(medical_df,
				x='children',
				y='charges',
				title='Number of children vs. Charges')
load_plot(fig)

# Examine relationship of the values in different columns using correlation coefficient

print(f"Correlation coefficient between charges and age: {medical_df.charges.corr(medical_df.age)}")
print(f"Correlation coefficient between charges and bmi: {medical_df.charges.corr(medical_df.bmi)}")
print(f"Correlation coefficient between charges and children: {medical_df.charges.corr(medical_df.children)}")


# To compute correlation between categorical values, they must first be converted into numeric

smoker_values = {'no': 0, 'yes': 1}
smoker_numeric = medical_df.smoker.map(smoker_values)
print(f"Correlation coefficient between charges and smoking: {medical_df.charges.corr(smoker_numeric)}")

# Make correlation matrix and display it in heat map

# First select only numerical values in the df

numeric_df = medical_df.select_dtypes(include=['number'])
print(numeric_df.corr())

# Build heatmap to visualize corelation matrix

plt.figure(figsize=(10, 8))
ax = sns.heatmap(numeric_df.corr(), cmap="Reds", annot=True)
plt.title('Corelation Matrix')
# plt.savefig('correlation_matrix.png')

# 2. Linear regression using single feature

# Estimate the value of charges using value age for non-smokers

# Filter only non-smokers
non_smoker_df = medical_df[medical_df.smoker == 'no']

# Plot the data

plt.figure(figsize=(10, 8))
plt.title('Age vs. Charges')
sns.scatterplot(data=non_smoker_df, x='age', y='charges', alpha=0.7, s=15)
# plt.savefig('age_vs_charges_non_smokers.png')

# Linear regression model for charges and age
# Formula of the model charges = w * age + b
# Define helper function estimate_charges to compute charges, given age, w and b


def estimate_charges(age, w, b):
	return w * age + b


w = 50
b = 100
ages = non_smoker_df.age
estimated_charges = estimate_charges(ages, w, b)

# Plot the estimated charges

plt.figure(figsize=(10, 8))
plt.plot(ages, estimated_charges, 'r-o')
plt.xlabel('Age')
plt.ylabel('Estimated Charges')
# plt.savefig('age_vs_estimated_charges.png')

# Plot model data vs real data

plt.figure(figsize=(10, 8))
target = non_smoker_df.charges
plt.plot(ages, estimated_charges, 'r', alpha=0.9)

plt.scatter(ages, target, s=8, alpha=0.8)
plt.xlabel('Age')
plt.ylabel('Charges')
plt.legend(['Estimate', 'Actual'])
# plt.savefig('age_charges_estimate_vs_real.png')


# Lets create helper function which takes w and b as inputs and create the above plot.

def try_parameters(w, b):
	ages = non_smoker_df.age
	target = non_smoker_df.charges
	estimated_charges = estimate_charges(ages, w, b)

	plt.figure(figsize=(10, 8))
	plt.plot(ages, estimated_charges, 'r', alpha=0.9)
	plt.scatter(ages, target, s=8, alpha=0.8)
	plt.xlabel('Age')
	plt.ylabel('Charges')
	plt.legend(['Estimate', 'Actual'])
	name_of_plot = f"age_charges_estimate_vs_actual_w{w}_b{b}.png"
	plt.savefig(name_of_plot)
	loss = rmse(targets, estimated_charges)
	print('RMSE Loss: ', loss)


targets = non_smoker_df.charges

predictions = estimate_charges(ages, w, b)



# Lost/Cost Function
# Function to calculate root mean square error(RMSE)

def rmse(targets, prediciotns):
	return np.sqrt(np.mean(np.square(targets - prediciotns)))

w = 310
b = -3750

result = rmse(targets, predictions)

# On average each element in prediction differs from actual target by 'result'= 8461 for w = 50 and b = 100

try_parameters(310, -3750)

# Linear regression using scikit-learn

model = LinearRegression()
help(model.fit)

inputs = non_smoker_df[['age']]
targets = non_smoker_df.charges

print('inputs shape:', inputs.shape)
print('targets shape:', targets.shape)

# Fit the model to the data

model.fit(inputs, targets)

# Test model predictions- the values in np.array are ages and the model returns predicted expenses
print(f"Test some predictions: {model.predict(np.array([[23], [37], [61]]))}")

# Test the model for the fill list of ages

predictions = model.predict(inputs)

print(rmse(targets, predictions))

# Display w and b of the model and visualize it using try_parameters function

print(f"Linear Regression model: \nw: {model.coef_}\nb: {model.intercept_}")
try_parameters(model.coef_[0], model.intercept_)
print('-' * 20)
# Build a model using SGDRegressor model using a stochastic gradient\

model_sgd = SGDRegressor()
model_sgd.fit(inputs, targets)
predictions_sgd = model_sgd.predict(inputs)
print(f"RMSE for SGD regression model:\n {rmse(targets, predictions_sgd)}")
print(f"SGDRegressor: w: {model_sgd.coef_}\nb: {model_sgd.intercept_}")
try_parameters(model_sgd.coef_[0], model_sgd.intercept_)

# Build Linear Regression model for smokers

smoker_df = medical_df[medical_df.smoker == 'yes']
inputs_smokers = smoker_df[['age']]
targets_smokers = smoker_df.charges

model_linear_smokers = LinearRegression()
model_linear_smokers.fit(inputs_smokers, targets_smokers)


predictions_smokers = model_linear_smokers.predict(inputs_smokers)
print('-' * 20)
print(f"RMSE for Linear Regression model with smokers data: {rmse(targets_smokers, predictions_smokers)}")
print(f"w = {model_linear_smokers.coef_[0]}\n b = {model_linear_smokers.intercept_}")