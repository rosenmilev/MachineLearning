from urllib.request import urlretrieve
import pandas as pd
import plotly.express as px
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

# medical_charges_url = 'https://raw.githubusercontent.com/JovianML/opendatasets/master/data/medical-charges.csv'
#
#
# urlretrieve(medical_charges_url, 'medical.csv')

# Create pandas data frame
medical_df = pd.read_csv('medical.csv')

# Examine data quality
# print(medical_df.describe())

# Some style settings of plotting libraries

sns.set_style('darkgrid')
matplotlib.rcParams['font.size'] = 14
matplotlib.rcParams['figure.figsize'] = (10, 6)
matplotlib.rcParams['figure.facecolor'] = '#00000000'

# Age column

print(medical_df.age.describe())

# Show histogram of age

fig = px.histogram(medical_df,
				   x='age',
				   # marginal='box',
				   nbins=47,
				   title='Distribution of Age')
fig.update_layout(bargap=0.1)
# fig.show()

# BMI column

fig = px.histogram(medical_df,
				   x='bmi',
				   marginal='box',
				   color_discrete_sequence=['red'],
				   title='Distribution of BMI ')
fig.update_layout(bargap=0.1)
fig.show()

# Charges column
# Smoker/Non-smoker
fig = px.histogram(medical_df,
				   x='charges',
				   marginal='box',
				   color='smoker',
				   color_discrete_sequence=['green', 'grey'],
				   title='Annual Medical Charges By Smokers/Non-Smokers'
				   )

fig.update_layout(bargap=0.1)
fig.show()

# Sex

fig = px.histogram(medical_df,
				   x='charges',
				   marginal='box',
				   color='sex',
				   color_discrete_sequence=['blue', 'red'],
				   title='Annual Medical Charges By Sex')
fig.update_layout(bargap=0.1)
fig.show()