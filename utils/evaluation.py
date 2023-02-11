import numpy as np
import time
import datetime
def DCG(sampled_rankings, q_doc_weights,rank_weights):
    """
    This funciton return the DCG.
    """
    cutoff = sampled_rankings.shape[1]
    return np.sum(
                q_doc_weights[sampled_rankings]*rank_weights[None, :cutoff],
            axis=1)

def NDCG_based_on_samples(sampled_rankings,q_doc_weights,rank_weights,cutoff):
    """
    This funciton return the NDCG based multiple samples for the same query.
    """
    n_samples=sampled_rankings.shape[0]
    ranklistLength=sampled_rankings.shape[1]
    assert ranklistLength>=cutoff,"The length of a rank list should be greater than the NDCG cutoff"
    if q_doc_weights.sum()<=0:
      return np.zeros(n_samples)
    ideal_ranking=np.argsort(-q_doc_weights)[:cutoff][None,:]
    dcg=DCG(sampled_rankings[:,:cutoff],
        q_doc_weights,rank_weights)
    idcg=DCG(ideal_ranking,
        q_doc_weights,rank_weights)
    return dcg/idcg
def NDCG_based_on_single_sample(sampled_ranking,q_doc_weights,rank_weights,cutoff):
    """
    This funciton return the NDCG based a sinlge samples for a query.
    """
    sampled_ranking=sampled_ranking[None,:]
    NDCG=NDCG_based_on_samples(sampled_ranking,q_doc_weights,rank_weights,cutoff)[0]
    return NDCG
# def NDCG_based_on_datasplit(sampled_ranking,q_label_vector,rank_weights,cutoff):
#     """
#     This funciton return the NDCG based qid in a datasplit.
#     """
#     NDCG=NDCG_based_on_single_sample(sampled_ranking,q_label_vector,rank_weights,cutoff)
#     return NDCG
def Update_NDCG_multipleCutoffsDatasplit(sampled_ranking,qid,dataSplit,rank_weights,cutoffs,NDCGDict):
    """
    This funciton return the NDCG @ different cutoffs based qid in a datasplit.
    """
    q_label_vector=dataSplit.query_values_from_vector(qid,dataSplit.label_vector)
    for cutoff in cutoffs:
        dataSplitName=dataSplit.name
        nameCur="_".join([dataSplitName,"NDCG",str(cutoff)])
        NDCGCur=NDCG_based_on_single_sample(sampled_ranking,q_label_vector,rank_weights,cutoff)
        NDCGDict[nameCur].append(NDCGCur)
def Update_NDCG_multipleCutoffsData(sampled_ranking,q_label_vector,rank_weights,cutoffs,NDCGDict):
    """
    This funciton return the NDCG @ different cutoffs.
    """
    for cutoff in cutoffs:
        nameCur="_".join(["NDCG",str(cutoff)])
        NDCGCur=NDCG_based_on_single_sample(sampled_ranking,q_label_vector,rank_weights,cutoff)
        NDCGDict[nameCur].append(NDCGCur)

def dicounted_metrics(metrics,gamma=0.995):
    """
    This funciton returns the discounted cumulative sum.
    """  
    m=len(metrics)
    results=np.zeros(m)
    previous_sum=0
    for i in range(m):
        previous_sum=previous_sum*gamma+metrics[i]
        results[i]=previous_sum
    return results
def outputNDCG(NDCGDict,OutputDict):
    """
    This funciton orgainze NDCG results.
    """    
    for key,value in NDCGDict.items():
        OutputDict[key+"_cumu"].append(dicounted_metrics(value)[-1])
        OutputDict[key+"_aver"].append(np.mean(value))
def disparity(exposure,rel,**kwargs):
    """
    This is an implementation of exposure disparity of a single query defined in Eq.29&30 in the following paper, 
    """    
    q_n_docs = rel.shape[0]
    swap_reward = exposure[:,None]*rel[None,:]
    q_result = np.sum((swap_reward-swap_reward.T)**2.)/(q_n_docs*(q_n_docs-1))
    return q_result
def disparityDivideFreq(exposure,rel,q_freq,**kwargs):
    """
    This is an implementation of exposure disparity of a single query defined in Eq.29&30 in the following paper, 
    """    
    q_result = disparity(exposure/q_freq,rel)
    return q_result
def L2(exposure,rel,**kwargs):
    """
    This function gives the l2 distance between two distribution.
    """  
    exposure=exposure/exposure.sum()
    rel=rel/rel.sum()
    return np.sum(np.abs(exposure-rel))/2
def evaluate_exposure_unfairness(data_split,fcn):
    """
    This is an implementation of exposure disparity of multiple queries defined in Eq.29&30 in the following paper, 
    Computationally Efficient Optimization ofPlackett-Luce Ranking Models for Relevance and Fairness. Harrie Oosterhuis SIGIR 2021 
    """
    queriesList=data_split.queriesList
    unfairness_list=[]
    for qid in queriesList:
        q_exposure=data_split.query_values_from_vector(qid,data_split.exposure)
        q_rel=data_split.query_values_from_vector(qid,data_split.label_vector)
        q_freq=data_split.query_freq[qid]
        if q_freq<=0 or q_exposure.sum()<=0 or q_rel.sum()<=0:
            continue
        unfairness_list.append(fcn(q_exposure,q_rel,q_freq=q_freq))
    unfairness_list=np.array(unfairness_list)
    return np.mean(unfairness_list)

def outputFairnessDatasplit(data,OutputDict):
    """
    This funciton orgainze fairness results.
    """   
    UnfairnessFcns={"disparity":disparity,"disparityDivideFreq":disparityDivideFreq,"L2":L2}
    dataSplits=[data.train,data.validation,data.test]
    for dataSplit in dataSplits:
        for fcnName,fcn in UnfairnessFcns.items():
            unfairness=evaluate_exposure_unfairness(dataSplit,fcn)
            logName="_".join([dataSplit.name,fcnName])
            OutputDict[logName].append(unfairness)
def outputFairnessData(data,OutputDict):
    """
    This funciton orgainze fairness results.
    """   
    UnfairnessFcns={"disparity":disparity,"disparityDivideFreq":disparityDivideFreq,"L2":L2}
    for fcnName,fcn in UnfairnessFcns.items():
        q_exposure=data.exposure
        q_rel=data.TrueAverRating
        q_freq=data.queryFreq
        unfairness=fcn(q_exposure,q_rel,q_freq=q_freq)
        logName=fcnName
        OutputDict[logName].append(unfairness)

class TimeExecute:
    def __init__(self):
        self.start=time.time()
        self.Allstart=self.start
        now = datetime.datetime.now()
        print(now)
    def getTime(self):
        CurTime=time.time()
        timeElapse=CurTime-self.start
        self.start=CurTime
        return timeElapse,CurTime-self.Allstart
 