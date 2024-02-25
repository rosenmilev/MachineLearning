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


ax = sns.heatmap(numeric_df.corr(), cmap="Reds", annot=True)
plt.title('Corelation Matrix')
plt.savefig('correlation_matrix.png')