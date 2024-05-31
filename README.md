# Used Car Price Prediction

Due to a decline in car manufacturing, the demand for used cars has surged, creating an exciting opportunity to explore what drives their pricing. This project aims to identify key factors affecting used car prices, such as body type, damage, and historical sales data, and develop a model to predict prices accurately. By integrating both visual and data-driven aspects, this project will help consumers make informed purchases and enable retailers to optimize their inventory, ensuring everyone gets the best value.

### Dataset

The dataset is from Kaggle- [US used car dataset](https://www.kaggle.com/datasets/ananaymital/us-used-cars-dataset)

The datasets represent the details of used cars in the different states of the United States of America.
License: Data files from the "US Used Cars Dataset" on Kaggle, © Original Authors. This dataset is for academic, research, and individual experimentation only and is not intended for commercial purposes.

#### Dataset Description

An original dataset of around 10GB with 66 columns was provided. Sampling was performed, resulting in a subset with features most relevant to price prediction, and all subsequent tasks utilize this sampled data. From summary statistics and data visualization, it was concluded that the dataset exhibits a positively skewed price distribution, indicating relatively few items with very high prices compared to the majority. Dataset cleaning was conducted to remove irrelevant data, columns with null values, and duplicate entries.

#### Dataset Cleaning

Jupyter Notebook: [Data Preprocessing](https://github.com/amruthapurnavadrevu/Used-Car-Price-Prediction/blob/main/Notebooks/CW_PreProcessing.ipynb)

- Filtered irrelevant data:
  - Excluded cars with confirmed salvage value because these vehicles typically have undergone significant damage, and their market values may not accurately reflect the standard pricing of non-damaged vehicles.
  - Excluded data corresponding to commercial vehicles since our focus is on passenger vehicles.
  - Aesthetic features, administrative features, and redundant attributes more effectively represented by other features were excluded.
  - Cars listed for sale before the year 2000 were excluded to focus on recent and contemporary vehicle models.
- Dropped the null values
  - The decision to drop null values from the dataset is a deliberate choice aimed at optimizing the accuracy of our used car price prediction model. The dataset encompasses a diverse range of cars, spanning from affordable low-end models to luxurious vehicles. In this heterogeneous landscape, each car's unique set of features plays a crucial role in determining its market value.
- Units were excluded to ensure the numerical uniformity of the data.
- Duplicates were excluded from the data.
- A random sample comprising 10% of the entire cleaned dataset was selected to enhance computational feasibility.
- The price values in our dataset exhibit significant imbalance due to the diverse range of cars included, spanning from low-end to luxury vehicles. To mitigate the impact of this imbalance on our modeling process, we apply a logarithmic transformation to the prices. This helps alleviate the skewness in the distribution, ensuring a more balanced representation of price variations across the entire spectrum of cars in the dataset.
- Categorical variables were encoded using one-hot encoding.

-- This will show the dataset dimensions before and after cleaning --
|  | Before cleaning | After cleaning |
|-------------------|------------|-------------------|
| Rows | 3000040 | 92547 |
| Columns | 66 | 25 |

### Main Notebook

This is the link to the main notebook: [Used Car Price Prediction](https://github.com/amruthapurnavadrevu/Used-Car-Price-Prediction/blob/main/Notebooks/used_car_price_prediction.ipynb)

### Clustering

[K-Means Clustering](https://github.com/amruthapurnavadrevu/Used-Car-Price-Prediction/blob/main/Notebooks/kmeans_clustering.ipynb)

#### Experimental design

Since clustering is an unsupervised learning technique, and the data contains labels suitable for supervised methods, an alternative approach was taken. First, prices in the dataset were grouped into three categories: low, mid, and high range. Then, k-means clustering was applied to the independent features (excluding price and derived price categories) to identify logical clusters and their relation to price values. The elbow method was used to find the optimal value of k.

 <p align="center">
    <img src="https://github.com/amruthapurnavadrevu/Used-Car-Price-Prediction/blob/main/Visualisations/kmeans_elbow.png" alt="Elbow Method for Number of Clusters" width="350"/>
</p>

To visualize the results, Principal Component Analysis (PCA) was used and data points were colored based on the clusters. To be able to compare the clustering results, PCA was used to visualize the dataset and color data points based on the price categories. Findings are shown in the Results section below.

##### Results

<p align="center">
    <img src="dhttps://github.com/amruthapurnavadrevu/Used-Car-Price-Prediction/blob/main/Visualisations/kmeans_cluster_viz.png" alt="Elbow Method for Number of Clusters" width="350"/>
    <img src="https://github.com/amruthapurnavadrevu/Used-Car-Price-Prediction/blob/main/Visualisations/kmeans_price_categ_viz.png" alt="Elbow Method for Number of Clusters" width="350"/>
</p>

It is evident that the mid-price category dominates all clusters. Consequently, due to the lack of clear separation between clusters based on price, further exploration into targeted techniques based on the formed clusters will not be pursued. However, the application of this technique allowed the distribution of the dataset to be observed.

##### Discussion

- Instead of using market knowledge to define price ranges, k-means clustering was employed to group the price data points.
- Principal Component Analysis (PCA) was utilized for visualization, reducing the data to two dimensions by expressing the principal components as a linear combination of original features to capture maximum variance.
- The used-car price prediction dataset contains labeled data with price as the dependent variable, making supervised learning the ideal approach.
- For clustering, all variables were encoded, and k-means clustering was performed on all columns except price and price_category.
- Analysis of the clusters revealed that the average price for most clusters falls into low or medium categories.
- This suggests clustering might not be useful for this task, as no distinct clusters were observed in the visualization.

### Decision Trees

[Link to notebook](https://github.com/amruthapurnavadrevu/Used-Car-Price-Prediction/blob/main/Notebooks/DecisionTrees_RandomForests.ipynb)

#### Experimental design

Input features: Feature importance was computed for all the features and the top 10 features with high feature importance were considered to train the decision tree and random forest models.

Output label: Price

#### Algorithms used

Since it is a supervised regression task, decision trees and random forest regressor algorithms were used.

##### Visualizations

1. Scatterplots with actual vs predicted price values were plotted to visualize how the models’ predictions fare compared to actual values. The plots were linear indicating that the predictions were close to the actual values.
 <p align="center">
    <img src="https://github.com/amruthapurnavadrevu/Used-Car-Price-Prediction/blob/main/Visualisations/Scatterplot_DecisionTrees.png" alt="Predicted vs Actual Values - Decision Trees" width="350"/>
    <img src="https://github.com/amruthapurnavadrevu/Used-Car-Price-Prediction/blob/main/Visualisations/Scatterplot_RandomForest.png" alt="Predicted vs Actual Values - Random Forest" width="350"/>
</p>

2. Learning curves for both decision trees and random forest regressor show if the models are underfitting or overfitting. In our case, the models are learning well from the training data. However, there is scope for improvement. This is a conscious choice since we’ve prioritized computational efficiency over performance.
<p align="center">
     <img src="https://github.com/amruthapurnavadrevu/Used-Car-Price-Prediction/blob/main/Visualisations/LearningCurves_DecisionTrees.png" alt="Learning Curve of Decision Tree Regressor" width="350"/>
     <img src="https://github.com/amruthapurnavadrevu/Used-Car-Price-Prediction/blob/main/Visualisations/LearningCurves_RandomForest.png" alt="Learning Curve of Decision Tree Regressor" width="350"/>
</p>

##### Results

| Model         | Data         | MSE    | R-squared score |
| ------------- | ------------ | ------ | --------------- |
| Decision Tree | Training Set | 0.0186 | 0.9438          |
|               | Test Set     | 0.0297 | 0.9115          |
| Random Forest | Training Set | 0.0074 | 0.9775          |
|               | Test Set     | 0.0208 | 0.9379          |

##### Hyperparameter Tuning

Hyperparameter tuning helps in identifying the optimal parameters that result in the best predictions for the model and helps the model to generalize better to unseen data, improving overall performance.

Randomized Search Cross-Validation (CV) was used for hyperparameter tuning of both decision trees and random forest regressors because it is more computationally efficient than an exhaustive grid search and provides a more manageable way to explore different combinations.

##### Discussion

For the regression task at hand, Random Forest Regressor outperformed Linear Regression and Decision Trees. The table below summarizes the performance of the three models:
| Model | MSE | R-squared score |
| ------------------ | ------- | -------------- |
| Linear Regression | 0.0505 | 0.8495 |
| Decision Trees | 0.0297 | 0.9115 |
| Random Forest | 0.0208 | 0.9379 |

Linear Regression assumes that the features and target variable have a linear relationship. However, for our dataset, the relationship is not perfectly linear. Therefore, Decision Trees and Random Forest Regressor have better performance since they can capture the complex patterns and interactions among features better. Random Forest is an ensemble learning method that builds multiple decision trees and combines their predictions. This leads to more robust and accurate predictions compared to individual decision trees.

The hyperparameter tuning has helped the performance because tuning helps find the parameters at which the model’s performance is optimized.

[Link to Linear Regression implementation](https://github.com/amruthapurnavadrevu/Used-Car-Price-Prediction/blob/main/Notebooks/LinearRegression.ipynb)


### Conclusion

In this coursework, we delve into two primary aspects: predicting used car prices. Clustering identifies data distribution and outliers, while decision trees identify key features that influence pricing. Our validated hypothesis asserts that the random forest regressor surpasses both decision trees and linear regression in predicting the price. 

The dataset is deemed realistic as it encompasses a diverse range of cars, spanning from low-end to luxury vehicles. The insights into key features influencing prices align with intuition, instilling confidence in the models' potential performance if deployed. However, it's crucial to acknowledge a trade-off between performance and computational efficiency in our models. While the accuracy may not be flawless, it is sufficiently high to provide valuable insights into the influential features and offer a reliable estimate of prices for a given set of features. If deployed, potential challenges related to computational efficiency could be mitigated by optimizing algorithms or leveraging parallel processing capabilities, ensuring a balance between accuracy and efficiency.