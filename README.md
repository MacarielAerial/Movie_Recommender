# Movie_Recommender
This is a movie recommendation system built with Scikit Surprise for LBS MAM's Machine Learning for Big Data's assignment  
*A folder named 'data' is required to be present in this repo's parent directory*

## High-level Summary ##

**Technical Pre-requsites**  
The script requires scientific computation libraries Numpy and Pandas as well as a scalable machine learning library Turicreate to run.
Turicreate is a much more obscure library than Numpy and Pandas. I found this library online through this link: 
https://www.analyticsvidhya.com/blog/2018/06/comprehensive-guide-recommendation-engine-python/
Turicreate solves the problem of computing extremely large sparse matrix computations that Pandas is ill-equiped to handle. The closest alternative for Turicreate for smaller datasets is scikit-surprise, an off-shot of the popular machine learning library scikit learn.

**Programme Input**  
The script demands users to specify inside the source code the path towards raw data and the path for output data once it is generated.
The only parameter that the user is allowed to set in the source code is *user_based* which determines whether the recommendation system uses *user-based collaborative filtering* or *item-based collaborative filtering* for recommendations.
The script by default uses cosine similarity to compute the similarity matrix.
Programme Output:
The script produces a two-dimensional array with by default 10 recommended movies for each user. Each row in the output matrix corresponds with one user-movie instace with the score and the rank.

**General Architecture**  
The script is designed to have three layers of functions including low-level, mid-level and high-level functions.
Low-level functions (marked by the class *Aux*) package otherwise less readable codes under comprehensible function names to improve readability. For example, *aggregate_ratings(df_ratings, df_movies* under the class *Aux* is simply a Pandas groupby and aggregate function applied on both movie and rating datasets.
Mid-level functions are powerhouse functions that define the script's overall logic. The function names include words such as load, clean, model and output corresponding to their actual roles in the workflow. Low-level functions are naturally packaged into mid-level functions for readablity.
High-level function is the one function *exec(self)* under the class *RecSys* that provides an intuitive method for the user to call to obtain the final output.



