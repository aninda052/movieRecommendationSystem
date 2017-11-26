import numpy as np
import pandas as pd
from sklearn import cross_validation as cv

def cosineSimilarity(item1,item2):
	Sum=0.0
	item1_squre=0.0
	item2_squre=0.0
	for (i,j) in zip(item1,item2):
		Sum+=(i*j)
		item1_squre+=i**2
		item2_squre+=j**2
		#print Sum,item1_squre,item2_squre
	tmp=(item1_squre**0.5)*(item2_squre**0.5)
	if tmp==0.0 or sum==0.0:
		return 0.0
	else : 
		return float(Sum/tmp)




def itemSimilarity(itemsNum,item_item_mat):
	Len=itemsNum
	item_item_similarity=np.zeros((Len, Len))
	for item1 in range(Len):

		for item2 in range(item1,Len):
			tmp=cosineSimilarity(item_item_mat[item1],item_item_mat[item2])
		 	item_item_similarity[item1][item2]=tmp
		 	item_item_similarity[item2][item1]=tmp
	return item_item_similarity



def itemRating(user,item,item_simi,userItemMatrix):
	Sum=0.0
	tmp=0.0
	x=list(np.where(userItemMatrix[user]>0))
	for i in x[0]:
		item1=i
		simi=item_simi[item][item1]
		user_item1_Rating=userItemMatrix[user][item1]
		item1AvgRating=np.mean(userItemMatrix.T[item1])
		Sum+=simi*(user_item1_Rating-item1AvgRating)
		tmp+=np.abs(simi)
	if tmp==0.0 or sum==0.0:
		return 0.0
	else : 
		return (Sum/tmp)+np.mean(userItemMatrix.T[item])



def recommendation(userNum,itemNum,item_simi,userItemMatrix):
	result=np.zeros((userNum+1, itemNum+1))
	for user in range(userNum):
		print user
		for item in range(itemNum):
			#topItem=topneighbour[item]
			result[user+1][item+1]=itemRating(user,item,item_simi,userItemMatrix)

	return result
	



def Error(predicted_value, true_value):
	predicted=predicted_value[true_value.nonzero()].flatten()
    	true=true_value[true_value.nonzero()].flatten()
    	return mean_squared_error(predicted, true)**0.5



if __name__ == '__main__' :

	

	#Rating data reading 

	rat_col=['userId','itemId','rating','st']
	rating_data=pd.read_csv('../data/u.data',sep="\t",names=rat_col)

	n_users = rating_data.userId.unique().shape[0]
	n_items = rating_data.itemId.unique().shape[0]
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

	item_col= ['movieId','movieTitle','releaseDate','videoReleaseDate','IMDb URL','unknown','action','adventure',
              		'animation','children\'s','comedy', 'crime','documentary','drama','fantasy',
	 		'film-noir','horror','husical','mystery','romance','sci-fi','thriller','war','western']
	
	item_data=np.array(pd.read_csv('../data/u.item',sep='|',names=item_col))
	item_data=np.delete(item_data,(0,1,2,3,4),1)

			
	print "item_similarity"

	item_similarity= itemSimilarity(n_items,item_data)

#Item_Item Collaborative Filtering

	print "predicted_items"

	predicted_items=recommendation(n_users,n_items,item_similarity,userItemMat)
	
	print "Error Checking"

	
	
	print (Error(predicted_items,test_userItemMat))

