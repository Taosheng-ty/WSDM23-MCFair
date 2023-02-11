JOKE_THRESHHOLD = 2
import pandas as pd
import numpy as np
import json
import os

import surprise
sigmoid = lambda x: 1. / (1. + np.exp(-(x - 3) / 0.1))
GROUP_BOUNDARIES = [[-1,-0],[0,1]] #Boundaries for Left and Right
class Item:
    def __init__(self, polarity, quality=1, news_group = None, id=0):
        """
        Creates an Document/article
        Assigns a Group depending on polarity
        """
        self.p = polarity
        self.q = quality
        self.id = id
        if (GROUP_BOUNDARIES[0][0] <= polarity <= GROUP_BOUNDARIES[0][1]):
            self.g = 0
        elif (GROUP_BOUNDARIES[1][0] <= polarity <= GROUP_BOUNDARIES[1][1]):
            self.g = 1
        else:
            self.g = 2
        self.news_group = news_group
    def get_features(self):
        tmp = [0] * 3
        tmp[self.g] = 1
        # return np.asarray([self.p,self.q, self.p**2] + tmp)
        return np.asarray([self.p, self.q] + tmp)


class Movie:

    def __init__(self, id, group):
        self.id = id
        self.g = group
def load_news_data(seed=18):
    #MEDIA_SOURCES = ["ABC","AP", "BBC", "Bloomberg", "Breitbart","Buzzfeed","CBS","CNN","Conservative Tribune", "Daily Mail", "Democrazy Now", "Fox News", "Huffington Post", "Intercept", "Life News", "MSNBC", "National Review", "New York Times", "The American Conservative", "The Federalist", "The Guardian", "Washington Post", "WorldNetDaily"]
    MEDIA_SOURCES = ["Breitbart", "CNN",
                     "Daily Mail", "Fox News", "Huffington Post", "MSNBC",
                     "New York Times", "The American Conservative", "The Guardian", "WorldNetDaily"]
    df = pd.read_csv("data/InteractiveMediaBiasChart.csv",)
    df["Group"] = df.Source.astype("category").cat.codes
    df["Bias"] /= 40
    df["Quality"] /= 62

    selector = df["Source"].isin(MEDIA_SOURCES)
    df_small = df[selector]
    #TODO shorten data
    df_tiny = pd.DataFrame(columns=df_small.columns)
    #print("LA")
    for source in MEDIA_SOURCES:
        one_source = df["Source"] == source
        x = df[one_source].sample(3,random_state=seed)
        #print("Subsample:", x)
        df_tiny = df_tiny.append(x)
    df_tiny["Group"] = df_tiny.Source.astype("category").cat.codes
    df["Group"] = df.Source.astype("category").cat.codes

    return df, df_small, df_tiny


def load_news_items(n = 30, completly_random = False, n_left=None):
    data_full, data_medium, data_tiny = load_news_data()
    items = []

    if completly_random:
        for index, row in data_full.sample(n).iterrows():
            items.append(Item(row["Bias"], quality=1, id=index, news_group=row["Group"]))
    elif n_left is not None:
        c_left = 0
        for index, row in data_full.sample(frac=1).iterrows():
            if((row["Bias"]<0 and c_left < n_left ) or (row["Bias"]>0 and len(items) < n - (n_left -c_left))):
                items.append(Item(row["Bias"], quality=1, id=index, news_group=row["Group"]))
            if( len(items)>= n):
                return items

    else:
        for index, row in data_tiny.iterrows():
            items.append(Item(row["Bias"],quality=1, id=index, news_group=row["Group"]))

    return items


def define_genre(meta_data):

    #Defining Genres
    #Generates a List of Lists of Movies in each Genre
    genres = []
    movie_g_id = []
    for ge in meta_data["genres"]:
        movie_g_id.append([])
        for temp in eval(ge):
            movie_g_id[-1].append(temp["id"])
            if temp not in genres:
                genres.append(temp)
    g_idx = [g["id"] for g in genres]
    #Modify the Genres according to the Group Id.
    meta_data["genres"] = meta_data["genres"].map(lambda xx: [xxx["id"] for xxx in eval(xx)])
    return meta_data, g_idx

def select_companies(meta_data):
    # Selecting Companies
    # MGM, Warner Bros, Paramount, 20th Fox, Columbia (x2)
    # 5 Movie Companies with most user ratings
    selected_companies = [1, 2, 3, 4, 7, 8]
    comp_to_group = [0, 1, 2, 3, 4, 4]

    comp = meta_data["production_companies"].value_counts().index[selected_companies]
    comp_dict = dict([(x, comp_to_group[i]) for i, x in enumerate(comp)])
    meta_data = meta_data.astype({"id": "int"})
    meta_data = meta_data[meta_data["production_companies"].isin(comp)]

    return meta_data, comp_dict

def select_movies(ratings, meta_data, n_movies = 100, n_user= 10000):

    # Use the 100 Movies with the most ratings
    po2 = ratings["movieId"].value_counts()

    #Select the n_movies with highest variance
    var_scores = [np.std(ratings[ratings["movieId"].isin([x])]["rating"]) for x in po2.index[:(n_movies*3)]]
    var_sort = np.argsort(var_scores)[::-1]

    selected_movies = po2.index[var_sort[:n_movies]]
    #selected_movies =po2.index[:n_movies]
    ratings = ratings[ratings["movieId"].isin(selected_movies)]
    meta_data = meta_data[meta_data["id"].isin(selected_movies)]

    po = ratings["userId"].value_counts()

    ratings = ratings[ratings["userId"].isin(po.index[:n_user])] # remove users with less than 10 votes
    meta_data = meta_data[meta_data["id"].isin(ratings["movieId"].value_counts().index[:])]

    return ratings, meta_data

def get_user_features_genre(ratings, ratings_full, meta_data, n_user, g_idx):
    # Generate User Features (Mean rating on each movie Genre)
    n_user = len(ratings["userId"].unique())
    user_features = np.zeros((n_user, len(g_idx)))
    user_id_to_idx = dict(zip(sorted(ratings["userId"].unique()), np.arange(n_user)))
    temp = pd.merge(ratings_full[ratings_full["userId"].isin(ratings["userId"].unique())], meta_data, left_on="movieId",
                    right_on="id")
    for j, g_id in enumerate(g_idx):
        temp2 = temp[[g_id in x for x in temp["genres"]]]
        ids = [user_id_to_idx[x] for x in temp2["userId"].unique()]
        user_features[ids, j] = temp2.groupby('userId')["rating"].mean()

    return user_features

def get_ranking_matrix_incomplete(ratings, meta_data, n_user):

    user_id_to_idx = dict(zip(sorted(ratings["userId"].unique()), np.arange(n_user)))
    # Create a single Ranking Matrix, only relevance for rated movies
    # Leave it incomplete
    # ranking_matrix = np.zeros((n_user, n_movies))
    ranking_matrix = np.zeros((n_user, len(meta_data["id"])))
    movie_id_to_idx = {}
    movie_idx_to_id = []
    print(np.shape(ranking_matrix))
    for i, movie in enumerate(meta_data["id"]):
        movie_id_to_idx[movie] = i
        movie_idx_to_id.append(movie)
        single_movie_ratings = ratings[ratings["movieId"].isin([movie])]
        ranking_matrix[[user_id_to_idx[x] for x in single_movie_ratings["userId"]], i] = single_movie_ratings[
            "rating"]

    return ranking_matrix
def get_matrix_factorization(ratings, meta_data, n_user, n_movies):
    # Matrix Faktorization
    algo = surprise.SVD(n_factors=50, biased=False)
    reader = surprise.Reader(rating_scale=(0.5, 5))
    surprise_data = surprise.Dataset.load_from_df(ratings[["userId", "movieId", "rating"]],
                                                  reader).build_full_trainset()
    algo.fit(surprise_data)

    pred = algo.test(surprise_data.build_testset())
    print("MSE: ", surprise.accuracy.mse(pred))
    print("RMSE: ", surprise.accuracy.rmse(pred))

    ranking_matrix = np.dot(algo.pu, algo.qi.T)
    # ranking_matrix = np.clip(ranking_matrix, 0.5, 5)

    # movie_idx_to_id = [surprise_data.to_raw_iid(x) for x in movies_to_pick]
    movie_idx_to_id = [surprise_data.to_raw_iid(x) for x in range(n_movies)]
    features_matrix_factorization = algo.pu
    print("Means: ", np.mean(features_matrix_factorization), np.mean(algo.qi.T))
    print("Feature STD:", np.std(features_matrix_factorization), np.std(algo.qi))
    print("Full Matrix Shape", np.shape(ranking_matrix), "rankinG_shape", np.shape(ranking_matrix))

    return ranking_matrix, features_matrix_factorization, movie_idx_to_id

def fit_movie_data(n_movies=100, n_user=10000, n_company=5, movie_features="factorization", dataFolder ="data/MovieData"):
    """
    Preprocesses the Movie Dataset
    Generate Rating Matrices with Matrix Factoriation which contains probability of each item being relevant. 
    """
    SimulatedRatingPath=os.path.join(dataFolder,"SimulatedRating.npy")
    #Loading Meta Data from the Movies
    metadataFolder=os.path.join(dataFolder,"movies_metadata.csv")
    meta_data = pd.read_csv(metadataFolder)[["production_companies", "id", "genres"]]
    #Delete Movies with Date as ID
    meta_data = meta_data.drop([19730, 29503, 35587]) # No int id

    #Get Genres
    meta_data, g_idx = define_genre(meta_data)

    #Filter by Production Company to obtain 5 Groups
    meta_data, comp_dict = select_companies(meta_data)

    #Y = meta data from selected Companies

    #Load Ratings
    ratingPath=os.path.join(dataFolder,"ratings.csv")
    ratings_full = pd.read_csv(ratingPath)
    ratings = ratings_full[ratings_full["movieId"].isin(meta_data["id"])]
    ratings, meta_data = select_movies(ratings, meta_data, n_movies, n_user)

    #Complete Ranking Matrix
    ranking_matrix = get_ranking_matrix_incomplete(ratings, meta_data, n_user)
    full_matrix, features_matrix_factorization, movie_idx_to_id = get_matrix_factorization(ratings, meta_data, n_user, n_movies)
    #Add the real rating for already rated movies
    full_matrix[np.nonzero(ranking_matrix)] = ranking_matrix[np.nonzero(ranking_matrix)]

    if movie_features == "factorization":
        user_features = features_matrix_factorization
    else:
        user_features = get_user_features_genre(ratings,ratings_full, meta_data, n_user,g_idx)

    #Generate Probability Matrix
    #ranking_matrix = np.clip((full_matrix - 1) / 4, a_min=0, a_max=1)
    prob_matrix = sigmoid(full_matrix)

    groups = [comp_dict[meta_data[meta_data["id"].isin([x])]["production_companies"].to_list()[0]] for x in
            movie_idx_to_id]

    po = ratings["userId"].value_counts()
    po2 = ratings["movieId"].value_counts()
    print("Number of Users", len(po.index), "Number of Movies", len(po2.index))
    print("the Dataset before completion is", len(ratings) / float(n_user*n_movies), " filled")
    print("The most rated movie has {} votes, the least {} votes; mean {}".format(po2.max(), po2.min(), po2.mean()))
    print("The most rating user rated {} movies, the least {} movies; mean {}".format(po.max(), po.min(), po.mean()))

    #The list of groups contains all movies
    assert(np.shape(groups) == (n_movies,))
    np.save(SimulatedRatingPath, [prob_matrix,user_features, groups])
def simulate_news_data(dataFolder):
    """
    Get the prob_matrix of news data which contains probability of item being relevant to a user. 
    """
    #User normal distribution pdf without the normalization factor
    SimulatedRatingPath=os.path.join(dataFolder,"SimulatedRating.npy")
    items=load_news_items()
    users = np.asarray([sample_user_base(distribution="bimodal") for i in range(50000)])
    assert type(items) == list
    item_affs = np.asarray([x.p for x in items])
    item_quality = np.asarray([x.q for x in items])
    #Calculating the Affnity Probability for each Item, based user polarity and user Openness
    prob_matrix = np.exp(-(item_affs[None,:] - users[:,[0]])**2 / (2*users[:,[1]]**2))*item_quality
    user_features=np.array([[np.nan] for i in prob_matrix])
    groups=[x.g for x in items]
    np.save(SimulatedRatingPath, [prob_matrix,user_features, groups])
preprocessingFcn={"Movie":fit_movie_data,"News":simulate_news_data}
def binarize_rating(dataFolder,dataset_name="Movie",seed=0,rerun=False):
    """
    binarize the rating. prob_matrix contains probability of each item being relevant. 
    """
    BinarizeddRatingPath=os.path.join(dataFolder,"BinarizedRating_trial_{}.npy".format(seed))
    SimulatedRatingPath=os.path.join(dataFolder,"SimulatedRating.npy")
    if not os.path.exists(BinarizeddRatingPath) or rerun:
        preprocessingFcn[dataset_name](dataFolder=dataFolder)
    prob_matrix, user_features, groups = np.load(SimulatedRatingPath, allow_pickle=True)
    n_user,n_items=prob_matrix.shape[0],prob_matrix.shape[1]
    rng = np.random.default_rng(seed)    
    random_matrix = rng.random((n_user, n_items))
    binarizedRating=np.asarray(prob_matrix > random_matrix, dtype=np.float16)
    averagedRating=np.mean(binarizedRating,axis=0)
    np.save(BinarizeddRatingPath, [averagedRating,binarizedRating,user_features, groups])

def sample_item(binaryRating, user_features,seed=0):
    """
    Yielding a item according the random seed
    """
    while True:
        rng = np.random.default_rng(seed)    
        random_order = rng.permutation(np.shape(binaryRating)[0])
        for i in random_order:
            yield (binaryRating[i,:], user_features[i,:])
        #print("All user preferences already given, restarting with the old user!")
def getdataFolder(dataset_info_path="local_dataset_info.txt",dataset_name="Movie"):
    """
    This function return the data folder
    """
    with open(dataset_info_path) as f:
        all_info = json.load(f)
    print(all_info,dataset_name)
    set_info = all_info[dataset_name]
    
    return set_info["dataFolder"]
def load_data(dataset_name="Movie",dataset_info_path="local_dataset_info.txt",RandomSeed=0,rerun=False,relvance_strategy=None,NumDocMaximum=None):
    """
    load the data and build a item class
    """
    dataFolder=getdataFolder(dataset_info_path,dataset_name)
    AverRating, groups, DataGenerator=GetData(dataFolder =dataFolder,dataset_name=dataset_name,RandomSeed=RandomSeed,rerun=rerun,NumDocMaximum=NumDocMaximum)
    ItemsRankingInstance=ItemsRankingClass(AverRating, groups, DataGenerator,relvance_strategy)
    return ItemsRankingInstance

def GetData(dataFolder ="data/MovieData",dataset_name="Movie",RandomSeed=0,rerun=False,NumDocMaximum=None):
    """
    load the data and build a item class
    """
    BinarizeddRatingPath=os.path.join(dataFolder,"BinarizedRating_trial_{}.npy".format(RandomSeed))
    if not os.path.exists(BinarizeddRatingPath) or rerun:
        binarize_rating(dataFolder=dataFolder,dataset_name=dataset_name,seed=RandomSeed,rerun=rerun)
    AverRating, binaryRating, user_features, groups = np.load(BinarizeddRatingPath, allow_pickle=True)
    # print(binaryRating.shape,user_features.shape,len(groups))
    DataGenerator=sample_item(binaryRating[:,:NumDocMaximum], user_features,RandomSeed)
    return AverRating[:NumDocMaximum], groups, DataGenerator

class ItemsRankingClass:
    def __init__(self,AverRating,groups,DataGenerator,relvance_strategy):
        numDoc=AverRating.shape[0]
        self.TrueAverRating=AverRating
        self.queryFreq=0
        self.docFreq=np.zeros(numDoc)
        self.clicks=np.zeros(numDoc)
        self.exposure=np.zeros(numDoc)
        self.cacheLists=[]
        self.weightClicksAver=np.zeros(numDoc)
        self.weightClicksSum=np.zeros(numDoc)
        self.ClickSum=np.zeros(numDoc)
        self.weightCLicksMatrix=[]
        self.rankingHist=[]
        self.DataGenerator=DataGenerator
        self.groups=groups
        self.numDoc=numDoc
        self.userHist=[]
        self.relvance_strategy=relvance_strategy
    def getNumDoc(self):
        return self.numDoc
    def getEstimatedAverageRelevance(self,userFeature):
        if self.relvance_strategy=="TrueAverage":
            return self.TrueAverRating
        elif self.relvance_strategy=="EstimatedAverage":
            return self.weightClicksAver
        else:
            raise 
    def updateStatistics(self,clicks,ranking,positionBias,userFeature):
        np.add.at(self.docFreq,ranking,1)
        np.add.at(self.exposure,ranking,positionBias)
        np.add.at(self.weightClicksSum,ranking,clicks/positionBias)
        np.add.at(self.ClickSum,ranking,clicks)
        self.weightClicksAver=self.weightClicksSum/(self.docFreq+1e-12)
        self.weightCLicksMatrix.append(clicks/positionBias)
        self.rankingHist.append(ranking)
        self.userHist.append(userFeature)
        self.queryFreq+=1

def sample_user_base(distribution, alpha =0.5, beta = 0.5, u_std=0.3, BI_LEFT = 0.5):
    """
    Returns a User of the News Platform
    A user cosists of is Polarity and his Openness
    """
    if(distribution == "beta"):
        u_polarity = np.random.beta(alpha, beta)
        u_polarity *= 2
        u_polarity -= 1
        openness = u_std
        #std = np.random.rand()*0.8 + 0.2
    elif(distribution == "discrete"):
        #3 Types of user -1,0,1. The neutral ones are more open
        u_polarity = np.random.choice([-1,0,1])
        if(u_polarity == 0):
            openness = 0.85
        else:
            openness = 0.1
    elif(distribution == "bimodal"):
        if np.random.rand() < BI_LEFT:
            u_polarity = np.clip(np.random.normal(0.5,0.2,1),-1,1)[0]
        else:
            u_polarity = np.clip(np.random.normal(-0.5, 0.2, 1), -1, 1)[0]
        openness = np.random.rand()/2 + 0.05 #Openness uniform Distributed between 0.05 and 0.55
    else:
        print("please specify a distribution for the user")
        return (0,1)
    return np.asarray([u_polarity, openness])

