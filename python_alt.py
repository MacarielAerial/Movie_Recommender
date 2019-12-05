'''
This file contains a recomendation system built with Python
'''

# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import turicreate
from collections import defaultdict
from sklearn.model_selection import train_test_split

# Specify global variables
input_path = '../data/ml-latest/'
output_path = 'output/'
raw_file_names = ['movies.csv', 'ratings.csv']
user_based = False

class RecSys:
	'''
	Contains mid-level functions
	'''
	def __init__(self, input_path, output_path, raw_file_names, user_based, n_rec = 10, test_size = 0.08, pop_thres = 50, diagnosis = True):
		# Configure object
		self.diagnosis = diagnosis
		self.user_based = user_based
		self.n_rec = n_rec
		self.test_size = test_size
		self.pop_thres = pop_thres

		# Accept inputs
		self.input_path = input_path
		self.output_path = output_path
		self.raw_file_names = raw_file_names

		# Initiate placeholder variables
		self.df_movies = pd.DataFrame()
		self.df_ratings = pd.DataFrame()
		self.rating_matrix = pd.DataFrame()
		self.rating_matrix_alt = pd.DataFrame()
		self.train = pd.DataFrame()
		self.test = pd.DataFrame()
		self.train_sf = None
		self.test_sf = None
		self.rec = None

	def load_data(self):
		print('Loading data into memory...')
		df_movies = pd.read_csv(self.input_path + 'movies.csv')
		df_ratings = pd.read_csv(self.input_path + 'ratings.csv')
		self.df_movies = df_movies
		self.df_ratings = df_ratings
	
	def clean_data(self):
		print('Cleaning data for the recommendor system...')
		# Check for duplicates
		Aux.diagnosis(self.df_ratings, self.df_movies, self.diagnosis)

		# Drop unused columns
		self.df_ratings = Aux.delete_columns(self.df_ratings)

		# Check for dimensionality before mutating data
		print(self.df_movies.shape, self.df_ratings.shape)

		# Drop duplicates
		self.df_ratings, self.df_movies = Aux.delete_duplicates(self.df_ratings, self.df_movies)

		# Aggregate ratings
		self.df_ratings, self_df_movies = Aux.aggregate_ratings(self.df_ratings, self.df_movies)

		# Drop redundant movies
		self.df_movies = Aux.delete_redundants(self.df_movies, self.df_ratings)

		# Drop unpopular movies and inactive users
		vc2 = self.df_ratings.movieId.value_counts()
		self.df_ratings = self.df_ratings[self.df_ratings.isin(vc2.index[vc2.values > self.pop_thres])['movieId']]
		
		vc = self.df_ratings.userId.value_counts()
		self.df_ratings = self.df_ratings[self.df_ratings.isin(vc.index[vc.values > self.pop_thres])['userId']]

		# Check for dimensionality after mutating data
		print('Dimensionalities for rating data and movie data:')
		print(self.df_movies.shape, self.df_ratings.shape)

		# Glimpse the cleaned data
		print('Rating Data:')
		print(self.df_ratings.head())
		print('Movie Data')
		print(self.df_movies.head())

	def model(self):
		# Partition the data into train and test datasets
		# Convert the partitioned datasets into sparse dataframes
		print('Starting modelling procedure...')
		self.train, self.test, self.train_sf, self.test_sf = Aux.data_partition(self.df_ratings, test_size = self.test_size)

		# Configure the model of choice
		model = turicreate.item_similarity_recommender.create(self.train_sf, user_id = 'userId', item_id = 'movieId', target = 'rating', similarity_type = 'cosine')

		# make predictions on test dataset
		self.rec = model.recommend(users = self.test['userId'].unique().tolist(), k = self.n_rec)

		# Print out recommendations for 3 randomly selected users
		df = self.rec.to_dataframe()
		print('Sample recommendations for 3 randomly selected users') 
		print(df.loc[df['userId'].isin(np.random.choice(df['userId'].unique(), size = 3)),:])
		
	def model_alt(self):
		print('This function is still under construction')

	def output_data(self):
		# Save the dataframe into a csv file
		print('Saving all recommendations to memory')
		self.rec.save(output_path + 'top_' + str(self.n_rec) + '_rec.csv', format = 'csv')

	def exec(self):
		self.load_data()
		self.clean_data()
		self.model()
		self.output_data()

class Aux:
	'''
	Contains low-level functions
	'''
	def diagnosis(df_ratings, df_movies, diagnosis):
		# Check for duplicated movie ids
		n_uni_users = len(df_ratings['userId'].unique())
		n_uni_movies = len(df_movies['movieId'].unique())
		if diagnosis:
			print('The number of unique users in rating data is ' + str(n_uni_users))
			print('The number of unique movies in rating data is ' + str(n_uni_movies))
			print('The number of movies in movie data is ' + str(len(df_movies)))
	
	def delete_columns(df_ratings):
		# Drop unused column in rating data
		df_ratings.drop(['timestamp'], axis = 1, inplace = True)
		return df_ratings
		
	def delete_duplicates(df_ratings, df_movies):
		# Drop the same movie with different movie ids and their ratings
		duplicated_movie_id = df_movies.loc[df_movies.duplicated(subset = 'title', keep = 'first'), :]['movieId']
		df_movies = df_movies.loc[~df_movies['movieId'].isin(duplicated_movie_id), :]
		df_ratings = df_ratings.loc[~df_ratings['movieId'].isin(duplicated_movie_id), :]
		return df_ratings, df_movies

	def aggregate_ratings(df_ratings, df_movies):
		# Aggregate the same user's multiple ratings on the same movie by only keeping the highest rating
		df_ratings = df_ratings.groupby(['userId', 'movieId']).agg('max')
		# Reset index for both movies and ratings to allow interaction selection
		df_ratings.reset_index(inplace = True, drop = False)
		df_movies.reset_index(inplace = True, drop = True)
		return df_ratings, df_movies

	def delete_redundants(df_movies, df_ratings):
		# Drop movies where there are no ratings
		df_movies = df_movies.loc[df_movies['movieId'].isin(df_ratings['movieId']),:]
		return df_movies

	def data_partition(df_ratings, test_size):
		# Split the whole dataset into train and test data
		train, test = train_test_split(df_ratings, test_size = test_size)
		train_sf, test_sf = turicreate.SFrame(train), turicreate.SFrame(test)
		print('The number of unique users in training set is ' + str(len(train['userId'].unique())))
		print('The number of unique movies in training set is ' + str(len(train['movieId'].unique())))
		return train, test, train_sf, test_sf

	def get_top_n(predictions, n_rec):
		'''Deprecated'''
		# Map predictions to users
		top_n = defaultdict(list)
		for uid, iid, true_r, est, _ in predictions:
			top_n[uid].append((iid, est))

		# Sort the predictions for each user and retrieve the k highest ones
		for uid, user_ratings in top_n.items():
			user_ratings.sort(key = lambda x:x[1], reverse = True)
			top_n[uid] = user_ratings[:n_rec]
		return top_n
	
	def process_rec(rec, top_n, n_rec):
		'''Deprecated'''
		# Print 10 random predictions associated with users
		for uid, user_ratings in top_n.items():
			rec[uid] = [iid for (iid, _) in user_ratings]
		count = 0
		for uid in rec:
			if count <= 10:
				print(str(uid) + ' : ' + str(rec[uid]))
				count = count + 1
		# Save these recommendations to a csv file
		df_rec = pd.DataFrame.from_dict(rec, orient = 'index')
		# Sort the index
		df_rec.sort_index(inplace = True)
		# Re-index the column names
		df_rec.rename(columns={x:y for x,y in zip(df_rec.columns,range(1,len(df_rec.columns) + 1))}, inplace = True)
		return df_rec


def main():
	'''
	Contains high-level functions
	'''
	object = RecSys(input_path, output_path, raw_file_names, user_based)
	object.exec()







if __name__ == '__main__':
	main()
