# ML_UnsupervisedLearning_KMeansExample

'''
IN THIS EXAMPLE, 
- WE WILL USE THE KMEANS CLUSTER ALGORITHM TO ANALYZE BANKING CUSTOMER FEEDBACK DATA.
- OUR GOAL IS TO GAIN INSIGHT INTO OUR DEPOSITORS' PREFERENCES FOR INTERACTING WITH US
    > VIRTUALLY VIA OUR BANKING APP 
    > OR IN PERSON DIRECTLY WITH ONE OF OUR PERSONAL BANKERS
- USING THESE INSIGHTS WE CAN MAKE RECOMMENDATIONS TO OUR MANAGEMENT TEAM AS TO 
    > HOW TO BEST ALLOCATE OUR RESOURCES 
    > TO GIVE OUR CUSTOMERS THE CLIENT SERVICE EXPERIENCE THAT SUITS THEIR NEEDS
    > AND HOW WE CAN DO THIS MOST PROFITABLY 
'''

# Importing dependencies
import pandas as pd
from pathlib import Path
import hvplot.pandas
from sklearn.cluster import KMeans

# Reading in the CSV file as a Pandas DataFrame
service_ratings_df = pd.read_csv(
    Path("../Resources/service_ratings.csv")
)

# Reviewing the DataFrame
service_ratings_df.head()

# Visualizing with a scatter plot of the data
service_ratings_df.hvplot.scatter(x="mobile_app_rating", y="personal_banker_rating")

# Creating and initializing K-means model instance for 2 clusters
model = KMeans(n_clusters=2, random_state=1)

# Printing the model
model

# Fitting the data to the instance of the model
model.fit(service_ratings_df)

# Making predictions about the data clusters using the trained model
customer_ratings = model.predict(service_ratings_df)

# Printing the predictions
print(customer_ratings)

# Adding a column to the DataFrame that contains the customer_ratings information
service_ratings_df['customer rating'] = customer_ratings

# Reviewing the DataFrame
service_ratings_df.head()

# Plot the data points based on the customer rating
service_ratings_df.hvplot.scatter(
    x="mobile_app_rating", 
    y="personal_banker_rating", 
    by="customer rating"
)
