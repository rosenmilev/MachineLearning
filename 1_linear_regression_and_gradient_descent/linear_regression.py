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
fig.show()

# Examine relationship between charges and sex
plt.figure(figsize=(6, 8))
fig = sns.barplot(x='charges', y='sex', data=medical_df)
plt.savefig('bar_plot.png')
plt.show()