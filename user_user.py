import numpy as np
import pandas as pd
from sklearn import cross_validation as cv
from sklearn.metrics import mean_squared_error


def pearsonSimilarity(user1,user2):

	Sum=0.0
	user1_sum=0.0
	user2_sum=0.0
	user1_mean=np.mean(user1)
	user2_mean=np.mean(user2)
	user1_sum=0.0
	Len=len(user1)
	for (i,j) in zip(user1,user2):
		Sum+=((i-user1_mean)*(j-user2_mean))
		user1_sum+=(i-user1_mean)**2
		user2_sum+=(j-user2_mean)**2
		#print Sum,item1_squre,item2_squre

	tmp=((user1_sum/Len)**.5)*((user2_sum/Len)**.5)
	if tmp==0.0 or sum==0.0:
		return 0.0
	else : 
		return float(Sum/tmp)




def userSimilarity(usersNum,user_item_mat):
	Len=usersNum
	user_user_similarity=pd.DataFrame(index=range(Len),columns=range(Len))
	for user1 in range(Len):
		#print item1
		for user2 in range(user1,Len):
			tmp=scipy.stats.pearsonr(user_item_mat[user1],user_item_mat[user2])
		 	user_user_similarity[user1][user2]=tmp
		 	user_user_similarity[user2][user1]=tmp
	return user_user_similarity



def topNeighbours(userSimilarity,num_neighbours):
	users=userSimilarity.shape[0]
	neighbour=pd.DataFrame(index=range(num_neighbours),columns=userSimilarity.columns)
	for user in range(users):
		#print i
		x=userSimilarity[user]
		x=x.sort_values(ascending=False)
		a=x[:num_neighbours]
		b=x[:num_neighbours].index
		for j in range(num_neighbours):
			neighbour[user][j]=[a.iloc[j],b[j]]
	return neighbour



def itemRating(user,item,topUser,userItemMatrix):
	Sum=0.0
	tmp=0.0
	for u in topUser:
		#print i
		user1=u[1]
		simi=u[0]
		user1_item_Rating=userItemMatrix[user1][item]
		user1AvgRating=np.mean(userItemMatrix[user1])
		Sum+=simi*(user1_item_Rating-user1AvgRating)
		tmp+=np.abs(simi)
	return (Sum/tmp)+np.mean(userItemMatrix[user])



def recommendation(userNum,itemNum,topneighbour,userItemMatrix):
	result=np.zeros((userNum, itemNum))
	for user in range(userNum):
		print user
		for item in range(itemNum):
			topUsers=topneighbour[user]
			result[user][item]=itemRating(user,item,topUsers,userItemMatrix)

	return result
	

def Error(predicted_value, true_value):
	predicted=predicted_value[true_value.nonzero()].flatten()
    	true=true_value[true_value.nonzero()].flatten()
    	return mean_squared_error(predicted,true)**0.5

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
	
	print 'user_similarity'
	user_similarity= userSimilarity(n_users,userItemMat)

	num_neighbour=100
	print 'Neighbours'
	Neighbours=topNeighbours(user_similarity,num_neighbour)

	print 'predicted_items'
	predicted_items=recommendation(n_users,n_items,Neighbours,userItemMat)
	

	rows=test_data['userId'].shape[0]
	test_userItemMat=np.zeros((n_users, n_items))
	for i in range(rows):
		tmp=test_data.iloc[i]
		test_userItemMat[tmp[0]-1][tmp[1]-1]=tmp[2]
	
	print (Error(predicted_items,test_userItemMat))

	
	











	
	
