# Customer Segmentation using Unsupervised Machine Learning Techniques

This project applies unsupervised machine learning techniques to segment customers based on demographic and spending behavior data. The goal is to analyze the dataset and identify customer groups that can be targeted for personalized marketing strategies.

## Table of Contents
- [Overview](#overview)
- [Technologies Used](#technologies-used)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Challenges](#challenges)
- [Future Improvements](#future-improvements)
- [License](#license)

## Overview
The project uses **K-Means**, **DBSCAN**, and **Hierarchical Clustering** algorithms to group customers. By analyzing the clustering results, businesses can gain insights into different customer segments for targeted marketing campaigns.

### Project Steps:
1. **Data Preprocessing**: Clean and prepare the dataset for clustering.
2. **Exploratory Data Analysis (EDA)**: Analyze the dataset to understand the patterns and relationships.
3. **Modeling**: Apply K-Means, DBSCAN, and Hierarchical Clustering algorithms.
4. **Evaluation**: Analyze the results using evaluation metrics like silhouette scores.
5. **Visualization**: Use plots to visualize the clusters.

## Technologies Used
- **Python** - Programming language used for data processing and modeling.
- **Pandas** - Data manipulation and analysis library.
- **Numpy** - Numerical computing library for handling large data arrays.
- **Scikit-learn** - Machine learning library for clustering algorithms.
- **Matplotlib** - Plotting library for visualizing results.
- **Seaborn** - Data visualization library based on Matplotlib.

## Dataset
The dataset used for this project is a publicly available dataset from Kaggle, containing over 8,000 customer records. It includes demographic information (e.g., age, gender, income) and spending behavior (e.g., annual spend on products, online spending score).

- **Dataset Link**: [Kaggle Customer Dataset](https://www.kaggle.com/datasets)

## Installation

Follow these steps to run this project on your local machine:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/customer-segmentation.git
   
2. **Install dependencies**:
Navigate to the project directory and install required Python libraries:
```bash
cd customer-segmentation
pip install -r requirements.txt
```

3.**Download the dataset**:
Download the dataset from Kaggle and place it in the data/ directory of the project.

4. **Run the notebook**:
Open and run the Jupyter notebook to explore the data and apply the clustering algorithms.
```bash
jupyter notebook customer_segmentation.ipynb
```

5.**Usage**:
**Data Preprocessing**:
The notebook loads the dataset and performs data cleaning by handling missing values, encoding categorical variables, and scaling numerical data.
**Exploratory Data Analysis (EDA)**:
Various visualizations (histograms, scatter plots) are used to understand the distribution of customer features.
**Clustering**:
Three clustering algorithms are applied:
**K-Means Clustering**: Groups customers into a fixed number of clusters based on feature similarity.
**DBSCAN**: Density-based clustering that can detect clusters of varying shapes.
**Hierarchical Clustering**: Builds a dendrogram to show customer clusters in a hierarchical structure.

6. **Evaluation**:
Evaluation of clustering results is done using silhouette scores and visual inspection.

7. **Visualization**:
2D and 3D plots (using matplotlib and seaborn) are used to visualize the clusters formed by each algorithm.

8. **Results**
**K-Means Clustering**:
Produced reasonable results for well-separated clusters but struggled with irregularly shaped clusters.
**DBSCAN**:
Successfully identified dense clusters, especially useful for outliers, but parameter tuning is required for optimal performance.
**Hierarchical Clustering**:
Best-performing algorithm with clear customer segments when analyzed through a dendrogram.

9.**Challenges**
**Data Cleaning**: Missing values, outliers, and irrelevant features were handled before applying clustering algorithms.
**Parameter Tuning**: The challenge of choosing the right number of clusters in K-Means and setting the correct parameters for DBSCAN.
**Cluster Validation**: No ground truth labels were available, so silhouette scores and visual validation were used to evaluate the model's effectiveness.

10. **Future Improvements**:
Implement advanced feature engineering techniques to improve clustering accuracy.

Test additional clustering algorithms, such as Gaussian Mixture Models (GMM), to handle mixed data types.

Explore domain-specific features like customer behavior trends for further segmentation improvements.

**License**
This project is licensed under the MIT License - see the LICENSE file for details.










