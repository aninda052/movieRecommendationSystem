# movieRecommendationSystem
It's a small project for a ofline movie recommendation system.

Recommender systems typically produce a list of recommendations in one of two ways – through
collaborative filtering or through content-based filtering (also known as the personality-based approach).
Collaborative filtering approaches build a model from a user's past behaviour (items previously purchased or 
selected and/or numerical ratings given to those items) as well as similar decisions made by other users. 
This model is then used to predict items (or ratings for items) that the user may have an interest in.
Content-based filtering approaches utilize a series of discrete characteristics of an item in order to recommend 
additional items with similar properties.[9] These approaches are often combined (see Hybrid Recommender Systems).

There are 4 file .item_item.py and user_user.py file are for a Collaborative filtering base recommendation system .
The model was train over 100k data set.
recommendatinSystem.py file is for a Hybrid Recommender Systems .I use movie's genres for item similarities 
(Content-based filtering) and then use item-item Collaborative filtering approaches for prediction . The model was trained 
over 1M data set .
The last file recoSysFromScrach.py will build a Hybrid Recommender Systems from Scrach .
