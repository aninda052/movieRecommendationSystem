import numpy as np
import pandas as pd
from sklearn import cross_validation as cv
from sklearn.metrics import mean_squared_error



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




def itemSimilarity(itemsNum,user_item_mat):
	Len=itemsNum
	item_item_similarity=pd.DataFrame(index=range(Len),columns=range(Len))
	for item1 in range(Len):
		#print item1
		for item2 in range(item1,Len):
			tmp=cosineSimilarity(user_item_mat.T[item1],user_item_mat.T[item2])
		 	item_item_similarity[item1][item2]=tmp
		 	item_item_similarity[item2][item1]=tmp
	return item_item_similarity



def topNeighbours(itemSimilarity,num_neighbours):
	items=itemSimilarity.shape[0]
	neighbour=pd.DataFrame(index=range(num_neighbours),columns=itemSimilarity.columns)
	for i in range(items):
		print i
		x=itemSimilarity[i].sort_values(ascending=False)
		a=x[:num_neighbours]
		b=x[:num_neighbours].index
		for j in range(num_neighbours):
			neighbour[i][j]=[a.iloc[j],b[j]]
	return neighbour


def itemRating(user,item,topItem,userItemMatrix):
	Sum=0.0
	tmp=0.0
	for i in topItem:
		item1=i[1]
		#print '-'*10,item1
		simi=i[0]
		user_item1_Rating=userItemMatrix[user][item1]
		item1AvgRating=np.mean(userItemMatrix.T[item1])
		Sum+=simi*(user_item1_Rating-item1AvgRating)
		tmp+=np.abs(simi)

	if tmp==0.0 or sum==0.0:
		return 0.0
	else : 
		return (Sum/tmp)+np.mean(userItemMatrix.T[item])



def recommendation(userNum,itemNum,topneighbour,userItemMatrix):
	result=np.zeros((userNum, itemNum))
	for user in range(userNum):
		print user
		for item in range(itemNum):
			topItem=topneighbour[item]
			result[user][item]=itemRating(user,item,topItem,userItemMatrix)

	return result
	



def Error(predicted_value, true_value):
	predicted=predicted_value[true_value.nonzero()].flatten()
    	true=true_value[true_value.nonzero()].flatten()
    	return mean_squared_error(predicted, true)**0.5




if __name__ == '__main__' :

	name=['userId','itemId','rating','st']
	data=pd.read_csv('../data/data.csv',sep='\t',names=name)

	n_users = data.userId.unique().shape[0]
	n_items = data.itemId.unique().shape[0]
	userItemMat=np.zeros((n_users, n_items))

	train_data, test_data = cv.train_test_split(data, test_size=0.20)
	rows=train_data['userId'].shape[0]

	for i in range(rows):
		tmp=train_data.iloc[i]
		userItemMat[tmp[0]-1][tmp[1]-1]=tmp[2]

	print "item_similarity"
	item_similarity= itemSimilarity(n_items,userItemMat)
	
	print "Neighbours"
	Neighbours=topNeighbours(item_similarity,350)

	print 'predicted_items'
	predicted_items=recommendation(n_users,n_items,Neighbours,userItemMat)

	rows=test_data['userId'].shape[0]
	test_userItemMat=np.zeros((n_users, n_items))
	for i in range(rows):
		tmp=test_data.iloc[i]
		test_userItemMat[tmp[0]-1][tmp[1]-1]=tmp[2]
	
	print (Error(predicted_items,test_userItemMat))
	
	











	
	
