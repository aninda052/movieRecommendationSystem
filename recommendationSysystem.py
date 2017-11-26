import numpy as np
import pandas as pd
import scipy.spatial
from sklearn import cross_validation as cv
from sklearn.metrics import mean_squared_error
from math import sqrt






def itemSimilarity(itemsNum,data):
	item_similarity_cosine = np.zeros((itemsNum,itemsNum))
	for item1 in range(itemsNum):
		print item1
		for item2 in range(itemsNum):
			item_similarity_cosine[item1][item2] = 1-scipy.spatial.distance.cosine(data[item1],data[item2])

	return item_similarity_cosine



def predict(ratings, similarity):
        mean_user_item = ratings.mean(axis=0)
        ratings_diff = (ratings - mean_user_item[np.newaxis,:])
        pred = mean_user_item[np.newaxis,:] + ratings_diff.dot(similarity) / np.array([np.abs(similarity).sum(axis=1)])
	return pred

	



def rmse(prediction, ground_truth):
    prediction = prediction[ground_truth.nonzero()].flatten()
    ground_truth = ground_truth[ground_truth.nonzero()].flatten()
    return sqrt(mean_squared_error(prediction, ground_truth))



if __name__ == '__main__' :

	

	#Rating data reading 

	rat_col=['userId','itemId','rating','st']
	rating_data=pd.read_csv('../data/rating.csv',names=rat_col)

	n_users = 6040
	n_items = 3952
	userItemMat=np.zeros((n_users, n_items))

	#train_test split
	train_data, test_data = cv.train_test_split(rating_data, test_size=0.20)

	rows=train_data['userId'].shape[0]
	for i in range(rows):
		tmp=train_data.iloc[i]
		userItemMat[tmp[0]-1][tmp[1]-1]=tmp[2]


	rows=test_data['userId'].shape[0]
	test_userItemMat=np.zeros((n_users, n_items))
	for i in range(rows):
		tmp=test_data.iloc[i]
		test_userItemMat[tmp[0]-1][tmp[1]-1]=tmp[2]


#For Content-base Item Similarity

	data =open('../data/movie.csv','r')
	item_data=np.zeros((n_items,18))
	genre = {"action":0,"adventure":1,"animation":2,"children's":3,"comedy":4,"crime":5,"documentary":6, "drama":7,"fantasy":8,"film-noir":9,"horror":10,"musical":11,"mystery":12,"romance":13,"sci-fi":14,"thriller":15, "war":16,"western":17 }

	for row in data:
		tmp=row.split(',')
		item=int(tmp[0])-1
		gen=tmp[len(tmp)-1].split('|')
		for i in gen:
			i=genre[i.strip().lower()]
			item_data[item][i]=1
			

	print "item_similarity"

	item_similarity= itemSimilarity(n_items,item_data)
	if np.isnan(item_similarity).any()==True :
		np.nan_to_num(item_similarity,copy=False)

#Item_Item Collaborative Filtering

	print "predicted_items"

	item_prediction = predict(userItemMat, item_similarity)
	
	print "Error Checking"

	print 'Item-based CF RMSE: ' + str(rmse(item_prediction, test_userItemMat))

 

3.26


