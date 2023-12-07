#Principle Component Analysis

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
import plotly.express as px
import graphviz
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title='PCA Simulator', layout='centered')
st.title('Principle Component Analysis (PCA)') #TODO: add icon and header animation here
st.header('Description')

info_desc = """[📜 ref](https://en.wikipedia.org/wiki/Principal_component_analysis)\n
Principal component analysis (PCA) is a popular technique for analyzing large datasets containing a high number of dimensions/features per observation, 
increasing the interpretability of data while preserving the maximum amount of information, and enabling the visualization of multidimensional data. 
Formally, PCA is a statistical technique for reducing the dimensionality of a dataset. 
"""

markdown_desc = """Principal Component Analysis (PCA) is a dimensionality reduction technique used in linear algebra 
and statistics to simplify and explore the structure of high-dimensional data. It is particularly valuable for reducing the complexity of data 
while preserving as much of its variance as possible. PCA accomplishes this by transforming the original data into a new set of uncorrelated variables called principal components. 
Here's a description of PCA in linear algebra:"""



app_desc = """
- You can use this app to apply PCA to your dataset and see usefull information about embedded dataset **(PCA applied)**.  
- In following flow chart you can see the flow of the app when you upload your dataset.
- For performing PCA You have to upload your dataset.
- After performing PCA you can download pca applied dataset.
"""
st.info(info_desc)
st.markdown(markdown_desc)

# Define a list of titles and descriptions
sections = [
    {
        "title": "Data Representation",
        "description": "Suppose you have a dataset with n data points, each consisting of p features (dimensions). This dataset can be represented as an n x p matrix, where each row represents a data point and each column represents a feature."
    },
    {
        "title": "Centering the Data",
        "description": "The first step in PCA is to center the data by subtracting the mean of each feature from its respective values. This centers the data around the origin and removes any translation effects."
    },
    {
        "title": "Covariance Matrix",
        "description": "PCA calculates the covariance matrix of the centered data. The covariance matrix measures how the different features in the dataset vary together. It is a p x p matrix, where each entry (i, j) represents the covariance between feature i and feature j."
    },
    {
        "title": "Eigendecomposition",
        "description": "The next step is to perform an eigendecomposition (eigenvalue decomposition) of the covariance matrix. This decomposition breaks down the covariance matrix into a set of eigenvalues and corresponding eigenvectors. These eigenvectors represent the principal components of the data."
    },
    {
        "title": "Principal Components",
        "description": "The eigenvectors obtained from the eigendecomposition are the principal components of the data. These components are orthogonal to each other, meaning they are uncorrelated. The first principal component has the largest eigenvalue, the second principal component has the second-largest eigenvalue, and so on. Each principal component explains a portion of the total variance in the data."
    },
    {
        "title": "Dimension Reduction",
        "description": "To reduce the dimensionality of the data, you can select a subset of the top k principal components that capture a significant portion of the total variance (where k is typically much smaller than p). This reduces the data from p dimensions to k dimensions."
    },
    {
        "title": "Data Reconstruction",
        "description": "You can project the centered data onto the selected principal components to obtain a lower-dimensional representation of the original data. This is done by taking the dot product of the centered data with the chosen principal components."
    }
]
# Define the content for Data Preprocessing expansion
data_preprocessing_content = """
Data Preprocessing involves preparing the dataset for PCA analysis. Here are the key steps:

1. **Handle Missing Data:** Address missing or incomplete data points by imputing missing values or removing rows/columns with missing data.

2. **Feature Scaling:** Scale the features to ensure they have the same scale, typically using standardization or normalization.

3. **Outlier Detection:** Identify and handle outliers, which can significantly impact PCA results. **`We implemented it in next version of app.`**

4. **Data Transformation:** Apply transformations such as logarithmic or power transformations to improve data distribution. **`We implemented it in next version of app.`**

5. **End Data Preprocessing:** Complete the data preprocessing step before PCA analysis.
"""

# Create the PCA flowchart
pca_dot = graphviz.Digraph()
pca_dot.node('Start', 'Start')
pca_dot.node('CenterData', 'Center the Data')
pca_dot.node('CovarianceMatrix', 'Calculate Covariance Matrix')
pca_dot.node('Eigendecomposition', 'Eigendecomposition')
pca_dot.node('SelectPrincipalComponents', 'Select Principal Components')
pca_dot.node('DimensionReduction', 'Dimension Reduction')
pca_dot.node('DataReconstruction', 'Data Reconstruction')
pca_dot.node('End', 'End')

pca_dot.edges([
    ('Start', 'CenterData'),
    ('CenterData', 'CovarianceMatrix'),
    ('CovarianceMatrix', 'Eigendecomposition'),
    ('Eigendecomposition', 'SelectPrincipalComponents'),
    ('SelectPrincipalComponents', 'DimensionReduction'),
    ('DimensionReduction', 'DataReconstruction'),
    ('DataReconstruction', 'End')
])

# Create the Data Preprocessing flowchart
data_preprocessing_dot = graphviz.Digraph()
data_preprocessing_dot.node('Start', 'Start')
data_preprocessing_dot.node('HandleMissingData', 'Handle Missing Data')
data_preprocessing_dot.node('FeatureScaling', 'Feature Scaling')
data_preprocessing_dot.node('OutlierDetection', **{'label': 'Outlier Detection (Future Feature)', 'fontcolor': 'black', 'style': 'filled', 'fillcolor': 'green', 'color':'red'})
data_preprocessing_dot.node('DataTransformation', **{'label': 'Data Transformation (Future Feature)', 'fontcolor': 'black', 'style': 'filled', 'fillcolor': 'green', 'color':'red'})
data_preprocessing_dot.node('EndDataPreprocessing', 'End Data Preprocessing')
data_preprocessing_dot.edges([
    ('Start', 'HandleMissingData'),
    ('HandleMissingData', 'FeatureScaling'),
    ('FeatureScaling', 'OutlierDetection'),
    ('OutlierDetection', 'DataTransformation'),
    ('DataTransformation', 'EndDataPreprocessing')
])

pca_components_help = """
If 0 $$<$$ `Input Number` $$<$$ 1 system will select the number of components such that the amount of variance that needs to be explained is greater than the percentage specified by `Input Number`.  
If `Input Number` $$\geq$$ 1 then integer part of it will be Number of components to keep.
"""

def pre_processing_func(df):       
    # missing value removing
    df.dropna(axis=1, how='any', inplace=True)
    # feature scaling
    scaler = StandardScaler()
    df_new = scaler.fit_transform(df) # here we create df_new to use it in futuer to gradient decent of neural network but now we use sklearn
    return df_new
@st.cache_data
def convert_df(np_array):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    df = pd.DataFrame(np_array)
    return df.to_csv(index=False).encode('utf-8')

# Create expandable sections dynamically
for section in sections:
    with st.expander(f"**{section['title']}**"):
        st.write(section['description'])
# Create another expandable sections dynamically
st.write("Also we need some Data preprocessing before going through PCA procedure")
# Display the flowchart and expandable Data Preprocessing content in Streamlit
with st.expander("Data Preprocessing Details"):
    st.write(data_preprocessing_content)
# App Description
st.header('App Description')
st.write(app_desc)
# upload personal dataset
columns = None
st.warning('🤖 Keep in minds that if you have **not a number** columns you have to change them into number by appropriate method for each columns and then upload your file')
upload_files = st.sidebar.file_uploader(label='Upload your own data', help='You can upload your dataset here', type=['csv'])
if upload_files is not None:
    df = pd.read_csv(upload_files)
    df = df.select_dtypes(exclude='O') # this is for test i have to remove it late
    # preprocessing dataset
    columns = df.columns
    str_columns = df.select_dtypes(include='O').columns
    if len(str_columns) != 0:
        st.info('You have some Object type columns, please encode them to number')
    else:
        df = pre_processing_func(df=df)
        # create a form to get pca parameters from users
        with st.sidebar.form('Details'):
            pca_components = st.number_input('Number of Components', min_value=0.00, max_value=float(len(columns)), format="%.2f", help=pca_components_help)
            submitted = st.form_submit_button("Submit")
        if submitted:
            if pca_components >= 1:
                pca_components = int(pca_components)
            elif pca_components == 0:
                st.sidebar.warning('you can not set it to 0')
                st.stop()
            pca = PCA(n_components=pca_components)
            transform_df = pca.fit_transform(df)
            st.write('Breif show of your transformed dataset:')
            st.dataframe(transform_df[0:5])
            # TODO: add appropriate features nad plots to show several prespectives of PCA transformed data
            fig, ax = plt.subplots()
            cumulative_sum = np.cumsum(pca.explained_variance_ratio_)
            ax = sns.barplot(x=pca.get_feature_names_out().tolist(), y=cumulative_sum)
            plt.title('Cummulative of Variance', fontsize = 10)
            plt.grid(True)
            ax.set_yticks(np.linspace(0,1,len(cumulative_sum)))
            fig.set_figwidth(4)
            fig.set_figheight(1)
            plt.xticks(fontsize=5)
            plt.yticks(fontsize=5)
            st.pyplot(fig)
            # download PCA applied dataset
            csv = convert_df(transform_df)
            st.download_button(label="Download full data as CSV",data=csv,file_name='pca_applied_dataset.csv',mime='text/csv')
    

tab1, tab2 = st.tabs(["Data Preprocessing Flow Chart", "PCA Procedure Flow Chart"])

with tab1:
   st.header('PCA Procedure Flow Chart')
   st.graphviz_chart(pca_dot)

with tab2:
    st.header('Data Preprocessing Flow Chart')
    st.graphviz_chart(data_preprocessing_dot)
