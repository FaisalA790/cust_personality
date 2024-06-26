import streamlit as st
import pickle
import numpy as np
import pandas as pd
#import matplotlib as plt
#import seaborn as sns

# Load the model, PCA, and scaler using pickle
with open('kmeans_model.pkl', 'rb') as f:
    kmeans = pickle.load(f)

with open('pca.pkl', 'rb') as f:
    pca = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

st.title('Customer Personality Analysis')

# User input
year_of_birth = st.number_input('Year of Birth', min_value=1900, max_value=2024)
income = st.number_input('Income', min_value=0.0)
# Marital Status
marital_status_married = st.checkbox('Married')
marital_status_single = st.checkbox('Single')
marital_status_other = st.checkbox('Other')
# Education
education_graduate = st.checkbox('Graduate')
education_undergraduate = st.checkbox('Undergraduate')
education_postgraduate = st.checkbox('Postgraduate')

# Prepare the input data
user_input = pd.DataFrame({
    'Year of Birth': [year_of_birth],
    'Income': [income],
    'Marital Status_Married': [1 if marital_status_married else 0],
    'Marital Status_Single': [1 if marital_status_single else 0],
    'Marital Status_Other': [1 if marital_status_other else 0],
    'Education_Graduate': [1 if education_graduate else 0],
    'Education_Undergraduate': [1 if education_undergraduate else 0],
    'Education_Postgraduate': [1 if education_postgraduate else 0]
})

# Scale the user input
user_input_scaled = scaler.transform(user_input)

# Apply PCA transformation
user_input_pca = pca.transform(user_input_scaled)

# Predict cluster
cluster = kmeans.predict(user_input_pca)
st.write(f'The customer belongs to cluster: {cluster[0]}')

# Visualize clusters
#def plot_clusters(PCA_data, cluster_labels, user_input_pca, title):
#    plt.figure(figsize=(10, 6))
#    unique_labels = np.unique(cluster_labels)
#    palette = sns.color_palette("viridis", len(unique_labels))
#    
#    # Create a DataFrame for easy handling in seaborn
#    plot_data = pd.DataFrame(PCA_data, columns=['col1', 'col2', 'col3'])
#    plot_data['Cluster'] = cluster_labels

#    sns.scatterplot(x='col1', y='col2', hue='Cluster', palette=palette, data=plot_data, legend='full')
#    plt.scatter(user_input_pca[:, 0], user_input_pca[:, 1], c='red', marker='x', s=100)  # New customer point
#    plt.title(title)
#    plt.xlabel('PCA Component 1')
#    plt.ylabel('PCA Component 2')
#    plt.legend(title='Clusters')
#    st.pyplot(plt)

# Load your original data (replace with your actual data)
#original_data = pd.read_excel('marketing_campaign1 (1).xlsx')
#original_data_scaled = scaler.transform(original_data)
#PCA_data = pca.transform(original_data_scaled)
#cluster_labels = kmeans.predict(PCA_data)

#plot_clusters(PCA_data, cluster_labels, user_input_pca, 'K-Means Clustering')
