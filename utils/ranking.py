import numpy as np
from qpsolvers import solve_qp
import random
from mip import Model, xsum, minimize, BINARY
from itertools import permutations
import matplotlib.pyplot as plt
import scipy
import scipy.optimize as opt
import utils.birkhoff as birkhoff
import utils.simulation as sim
import utils.PLfair.utils.PLFairTrain as PLFairTrain
import utils.PLfair.utils.plackettluce as pl
def unfairnessDoc(obs,relevance_esti_orig,fairness_strategy,**kwargs):

    """
    return the exposure disparity of each documents according to different disparity strategy.
    """
    relevance_esti_orig=np.clip(relevance_esti_orig,0,np.inf)
    obs=obs
    relevance_esti=relevance_esti_orig     
    doc_num=obs.shape[0]
    if fairness_strategy=="FairCo": 
        ##please refer to https://arxiv.org/abs/2005.14713
        relevance_esti_clip=np.clip(relevance_esti,1e-2,np.inf)  ## to avoid zero.
        ratio = obs/relevance_esti_clip
        swap_reward=ratio[:,None]-ratio[None,:] ## Eq.10 in above paper
        unfairness = np.max(swap_reward,axis=0) ## the equation after Eq.16    

    elif fairness_strategy=="FairCo_maxnorm": 
      relevance_esti_clip=np.clip(relevance_esti,1e-2,np.inf)  ## to avoid zero.
      ratio = obs/relevance_esti_clip
      swap_reward=ratio[:,None]-ratio[None,:]
      unfairness = np.max(swap_reward,axis=0)
      unfairness=unfairness/(np.max(np.abs(unfairness))+1e-10)

    elif fairness_strategy=="FairCo_multip.": 
      swap_reward = obs[:,None]*relevance_esti[None,:]
      unfairness = np.max(swap_reward-swap_reward.T,axis=0)
      unfairness=unfairness/(np.max(np.abs(unfairness))+1e-10)

    elif fairness_strategy=="GradFair":
      swap_reward = obs[:,None]*relevance_esti[None,:]
      q_result = np.sum((swap_reward-swap_reward.T)*relevance_esti[:,None],axis=0)/(doc_num*(doc_num-1))
      unfairness=q_result
      unfairness=unfairness/(np.max(np.abs(unfairness))+1e-10)

    elif fairness_strategy=="FFC":
      exposure_quota=kwargs["exposure_quota"]
      exposure_left = exposure_quota-obs
      exposure_feasible_id=exposure_left>0
      relevance_esti_rank= np.copy(relevance_esti)
      relevance_esti_rank[exposure_feasible_id]=relevance_esti[exposure_feasible_id]+10
      unfairness=relevance_esti_rank

    elif fairness_strategy=="FFC_v1":
      exposure_quota=kwargs["exposure_quota"]
      exposure_left = (exposure_quota-obs)
      unfairness=exposure_left
    else:
      raise     
    return unfairness

def multiple_rankings(scores, rankListLength):
    """
    This function gives multiple ranking lists according to the scores, and the length of each ranking list is rankListLength.
    """
    n_samples = scores.shape[0]
    n_docs = scores.shape[1]
    rankListLength = min(n_docs, rankListLength)
    ind = np.arange(n_samples)
    rankingScore=-scores
    if n_docs==rankListLength:
      rankings=np.argsort(rankingScore,axis=1)
      return rankings
    partition = np.argpartition(rankingScore, rankListLength, axis=1)[:,:rankListLength]
    sorted_partition = np.argsort(rankingScore[ind[:, None], partition], axis=1)
    rankings = partition[ind[:, None], sorted_partition]
    return rankings
def single_ranking(score, rankListLength):
    """
    This function gives a ranking according to the score, and the length of the ranking list is rankListLength.
    """
    ranking=multiple_rankings(score[None,:],rankListLength)[0]
    return ranking
def normalize(*param):
    """
    This function normalize all element in param.
    """
    # for i in param:
    #     assert i is isinstance(np)
    return [ i/i.sum() for i in param]
def LP(positionBias,popularity,ind_fair=True, group_fair=False, debug=False,\
     w_fair = 1, group_click_rel = None, impact=True, LP_COMPENSATE_W=10):
    n = popularity.shape[0]
    G = np.arange(n)
    n_g, n_i = 0, 0
    if(group_fair):
        n_g += (len(G)-1)*len(G)
    if(ind_fair):
        n_i += n * (n-1)

    n_c = n**2 + n_g + n_i


    c = np.ones(n_c)
    c[:n**2] *= -1
    c[n**2:] *= w_fair
    A_eq = []
    #For each Row
    for i in range(n):
        A_temp = np.zeros(n_c)
        A_temp[i*n:(i+1)*n] = 1
        assert(sum(A_temp)==n)
        A_eq.append(A_temp)
        c[i*n:(i+1)*n] *= popularity[i]

    #For each coloumn
    for i in range(n):
        A_temp = np.zeros(n_c)
        A_temp[i:n**2:n] = 1
        assert(sum(A_temp)==n)
        A_eq.append(A_temp)
        #Optimization
        c[i:n**2:n] *= positionBias[i]
    b_eq = np.ones(n*2)
    A_eq = np.asarray(A_eq)
    bounds = [(0,1) for _ in range(n**2)] + [(0,None) for _ in range(n_g+n_i)]


    A_ub = []
    b_ub = np.zeros(n_g+n_i)
    if(group_fair):
        U = []
        for group in G:
            #Avoid devision by zero
            u = np.max([sum(np.asarray(popularity)[group]), 0.01])
            U.append(u)
        comparisons = list(permutations(np.arange(len(G)),2))
        j = 0
        for a,b in comparisons:
            f = np.zeros(n_c)
            if len(G[a]) > 0 and len(G[b])>0:
                for i in range(n):
                    if impact:
                        tmp1 = popularity[i] / U[a] if i in G[a] else 0
                        tmp2 = popularity[i] / U[b] if i in G[b] else 0
                    else:
                        tmp1 = 1. / U[a] if i in G[a] else 0
                        tmp2 = 1. / U[b] if i in G[b] else 0
                    f[i*n:(i+1)*n] =  (tmp1 - tmp2) # * popularity[i] for equal impact instead of equal Exposure
                for i in range(n):
                    f[i:n**2:n] *= positionBias[i]
                f[n**2+j] = -1
                if group_click_rel is not None:
                    b_ub[j] = LP_COMPENSATE_W * (group_click_rel[b] - group_click_rel[a])
            j += 1
            A_ub.append(f)

    if(ind_fair):
        comparisons = list(permutations(np.arange(len(popularity)),2))
        j = 0
        for a,b in comparisons:
            f = np.zeros(n_c)
            if(popularity[a] >= popularity[b]):
                tmp1 = 1. / np.max([0.01,popularity[a]])
                tmp2 = 1. / np.max([0.01,popularity[b]])
                f[a*n:(a+1)*n] = tmp1
                f[a*n:(a+1)*n] *= positionBias
                f[b*n:(b+1)*n] = -1 *  tmp2
                f[b*n:(b+1)*n] *= positionBias

                f[n**2+n_g+j] = -1
            j += 1
            A_ub.append(f)
        
    res = opt.linprog(c, A_eq=A_eq, b_eq=b_eq, A_ub=A_ub, b_ub=b_ub, bounds=bounds, options=dict( tol=1e-5), method = "interior-point")
    probabilistic_ranking = np.reshape(res.x[:n**2],(n,n))


    if(debug):
        print("Shape of the constrains", np.shape(A_eq), "with {} items and {} groups".format(n, len(G)))
        print("Fairness constraint:", np.round(np.dot(A_eq,res.x),4))
        #print("Constructed probabilistic_ranking with score {}: \n".format(res.fun), np.round(probabilistic_ranking,2))
        print("Col sum: ", np.sum(probabilistic_ranking,axis=0))
        print("Row sum: ", np.sum(probabilistic_ranking,axis=1))
        #plt.matshow(A_eq)
        #plt.colorbar()
        #plt.plot()
        plt.matshow(probabilistic_ranking)
        plt.colorbar()
        plt.plot()

    #Sample from probabilistic ranking using Birkhoff-von-Neumann decomposition
    try:
        decomp = birkhoff.birkhoff_von_neumann_decomposition(probabilistic_ranking)
    except:
        decomp = birkhoff.approx_birkhoff_von_neumann_decomposition(probabilistic_ranking)

        if debug:
            print("Could get a approx decomposition with {}% accuracy".format(100*sum([x[0] for x in decomp])) )
            #print(probabilistic_ranking)
    return decomp
def sampleFromLp(decomp):
    """
    This function sample a ranking from Linear Programming (ILP) Method.
    """
    p_birkhoff = np.asarray([np.max([0, x[0]]) for x in decomp])
    p_birkhoff /= np.sum(p_birkhoff)
    sampled_r = np.random.choice(range(len(decomp)), 1, p=p_birkhoff)[0]
    ranking = np.argmax(decomp[sampled_r][1], axis=0)    
    return ranking

def ILP(positionBias,qRel,qExpVector,queryFreq,rankListLength,fairness_tradeoff_param):
    """
    This is the Integer Linear Programming (ILP) Method.
    """
    num_item=qExpVector.shape[0]
    positionBiasAllItem=np.zeros(num_item)
    positionBiasAllItem[:rankListLength]=positionBias
    positionBiasAllItem, qRel, qExpVector =normalize(positionBiasAllItem, qRel, qExpVector)
    # Objective : |A+w-(R+r)|. see Equation 3 in the following paper for details.
    # https://arxiv.org/pdf/1805.01788.pdf?ref=https://githubhelp.com 
    Obj=np.abs(qExpVector[:,np.newaxis]*queryFreq+positionBiasAllItem[np.newaxis,:]-(qRel*(queryFreq)+qRel)[:,np.newaxis])
    # print(qExpVector*queryFreq-(qRel*(queryFreq)+qRel))
    # print(positionBiasAllItem)
    model = Model()
    model.verbose=0
    x = [[model.add_var(var_type=BINARY) for j in range(num_item)] for i in range(num_item)] ## the decision variables
    model.objective = minimize( xsum(Obj[i][j]*x[i][j] for i in range(num_item) for j in range(num_item)))
    # constraint : only one element is 1 in each row and all other elements are 0.
    for i in range(num_item):
        model += xsum(x[i][j] for j in range(num_item) ) == 1
    # constraint : : only one element is 0 in each column and all other elements are 0.
    for j in range(num_item):
        model += xsum(x[i][j] for i in range(num_item) ) == 1  
    rel_argmax=(-qRel).argsort()[:rankListLength]
    mat_constr=qRel[:,np.newaxis]*positionBias[np.newaxis,:]
    idcg=np.sum(qRel[rel_argmax]*positionBias)
    model += xsum(mat_constr[i][j]*x[i][j] for i in range(num_item) for j in range(rankListLength))>=(1-fairness_tradeoff_param)*idcg
    model.optimize()
    xx=[]
    if model.num_solutions:
        for i, var in enumerate(model.vars):
            xx.append(var.x)
        xx=np.array(xx)
        xx=xx.reshape((num_item,num_item))
        rankingAllItem=np.argsort(-xx,0)[0,:]
        ranking=rankingAllItem[:rankListLength]
    #         print(xx,arg,"xx,arg")
    else:
        ranking=rel_argmax
    return ranking
def updateExposure(qid,dataSplit,ranking,positionBias):
    """
    This function update the exposure.
    """
    qExpVector=dataSplit.query_values_from_vector(qid,dataSplit.exposure)
    qExpVector[ranking]+=positionBias
# def getQuotaFromQP(relevance_esti,obs,FutureFairExpo,positionBias,NDCGconstraintParam=None):
#     """
#     This function calculate the additional quota needed for each document via qudratic optimization.
#     """
#     rankLength=positionBias.shape[0]
#     sum_sqaure_R=np.sum((relevance_esti)**2)
#     sum_mutiply_ER=np.sum(relevance_esti*obs)
#     B=obs*sum_sqaure_R-relevance_esti*sum_mutiply_ER
#     n_docs=relevance_esti.shape[0]
#     A=sum_sqaure_R*np.identity(n_docs)-relevance_esti[:,None]*relevance_esti[None,:]+np.identity(n_docs)*1e-2
#     lb=np.zeros(n_docs)
#     ub=np.ones(n_docs)*FutureFairExpo*positionBias[0]/positionBias.sum()
#     Constant_A=np.ones(n_docs)
#     G=None
#     h=None
#     if NDCGconstraintParam is not None:
#       G=-relevance_esti[None,:]
#       relevance_estiSorted=-np.sort(-relevance_esti)
#       n_futureSession=FutureFairExpo//positionBias.sum()
#       # print(n_futureSession,FutureFairExpo,"n_futureSession,FutureFairExpo")
#       h=-np.sum(relevance_estiSorted[:rankLength]*positionBias)*n_futureSession*(1-NDCGconstraintParam)
#       h=np.array([h])
#     x = solve_qp(A, B, G=G,h=h, A=Constant_A, b=FutureFairExpo,lb=lb,ub=ub)
#     if x is None:
#         return np.zeros(n_docs).astype(np.float)
#     return x

def extendDecisionVar(x,value,diagonal=False):
  if len(x.shape)==2 and diagonal:
    nDoc=x.shape[1]
    xExtend=np.identity(nDoc*2)
    xExtend[:nDoc,:nDoc]=x
    xExtend[nDoc:,nDoc:]*=value
  elif diagonal==False:
    nDoc=x.shape[-1]
    xExtend=np.ones_like(x)*value
    xExtend=np.concatenate([x,xExtend],axis=-1)  
  else:
    raise
  return xExtend


def getQuotaFromQP(relevance_esti,obs,FutureFairExpo,positionBias,\
  NDCGconstraintParam=None,exploration_tradeoff_param=0):
    """
    This function calculate the additional quota needed for each document via qudratic optimization.
    """
#     print(relevance_esti.dtype,obs.dtype,FutureFairExpo.dtype,positionBias.dtype)
    relevance_esti=np.clip(relevance_esti,1e-4,np.inf)
    rankLength=positionBias.shape[0]
    sum_sqaure_R=np.sum((relevance_esti)**2)
    sum_mutiply_ER=np.sum(relevance_esti*obs)
    B=obs*sum_sqaure_R-relevance_esti*sum_mutiply_ER
    n_docs=relevance_esti.shape[0]
    A=sum_sqaure_R*np.identity(n_docs)-relevance_esti[:,None]*relevance_esti[None,:]+np.identity(n_docs)*0.01
    lb=np.zeros(n_docs)
    ub=np.ones(n_docs)*FutureFairExpo*positionBias[0]/positionBias.sum()
    Constant_A=np.ones(n_docs)
    G=None
    h=None
    if NDCGconstraintParam is not None:
      G=-relevance_esti[None,:]
      relevance_estiSorted=-np.sort(-relevance_esti)
      n_futureSession=FutureFairExpo//positionBias.sum()
      h=-np.sum(relevance_estiSorted[:rankLength]*positionBias)*n_futureSession*(1-NDCGconstraintParam)
      h=np.array([h])
    # print(exploration_tradeoff_param)
    if exploration_tradeoff_param<=0:
      x = solve_qp(A, B, G=G,h=h, A=Constant_A, b=FutureFairExpo,lb=lb,ub=ub)
    else:
      A=extendDecisionVar(A,0.001,diagonal=True)
      B=extendDecisionVar(B,100)
      G=extendDecisionVar(G,0)
      Constant_A=extendDecisionVar(Constant_A,0)
      lb=extendDecisionVar(lb,0)
      ub=extendDecisionVar(ub,1e5)
      hexp=-(exploration_tradeoff_param-obs)
      Gexp=-np.concatenate([np.eye(n_docs),np.eye(n_docs)],axis=1)
      G=np.concatenate([G,Gexp],axis=0)
      h=np.concatenate([h,hexp],axis=0)
      # print(A.shape,B.shape)
      x = solve_qp(A, B, G=G,h=h, A=Constant_A, b=FutureFairExpo,lb=lb,ub=ub)
    if x is None:
#       print(relevance_esti,obs,FutureFairExpo,positionBias,NDCGconstraintParam,exploration_tradeoff_param)
      print("no solution")
      return FutureFairExpo/(relevance_esti.sum()+1e-10)*relevance_esti
    return x[:n_docs]

def getQuotaEachItemQuota(obs,q_rel,positionBias,n_futureSession,fairness_tradeoff_param): 
    """
    This funciton calcultes the Quota each item should get to keep fairness.
    """
    FutureFairExpo=positionBias.sum()*n_futureSession*fairness_tradeoff_param
    QuotaEachItem=getQuotaFromQP(q_rel,obs,FutureFairExpo,positionBias)
    # print(QuotaEachItem,q_rel/(q_rel.sum()+1e-5)*FutureFairExpo)
    return QuotaEachItem.astype(np.float)
def getQuotaEachItemNDCG(obs,q_rel,positionBias,n_futureSession,fairness_tradeoff_param,exploration_tradeoff_param=0): 
    """
    This funciton calcultes the Quota each item should get to keep fairness.
    """
    FutureFairExpo=positionBias.sum()*n_futureSession*1
    QuotaEachItem=getQuotaFromQP(q_rel,obs,FutureFairExpo,positionBias,\
      NDCGconstraintParam=fairness_tradeoff_param,exploration_tradeoff_param=exploration_tradeoff_param)
    # print(QuotaEachItem,q_rel/(q_rel.sum()+1e-5)*FutureFairExpo,QuotaEachItem.sum(),"QuotaEachItem")
    return QuotaEachItem.astype(np.float)
def getExpoBackwardCum(n_futureSession,rankListLength,positionBias):
    """
    This funciton calcultes the Exposure cumulation at each position backwards.
    """
    ExpoBackwardCum=np.zeros((n_futureSession,rankListLength))
    cum=0
    for j in range(rankListLength-1,-1,-1):
        for i in range(n_futureSession-1,-1,-1):
            cum=cum+positionBias[j]
            ExpoBackwardCum[i,j]=cum    
    return ExpoBackwardCum


def getVerticalRankingPartial(q_rel,rankListLength,n_futureSession,ExpoBackwardCum,QuotaEachItem,positionBias):
    """
    This funciton outputs ranklists by an vertical way.
    """
    Expo=np.zeros_like(q_rel)
    rankLists=[]
    n_doc=len(q_rel)
    for i in range(n_futureSession):
        rankLists.append([])
    for j in range(rankListLength):
        for i in range(n_futureSession):
            ranking=rankLists[i]
            q_relCur=q_rel+np.random.uniform(0,0.001,q_rel.shape) #to break tie
            q_relCur[ranking]=-np.inf
            if QuotaEachItem.sum()>=ExpoBackwardCum[i,j]:
                QuotaSatisfiedId=np.where(QuotaEachItem<=0)[0]
                mask=list(set(QuotaSatisfiedId.tolist()+ranking))
                if len(mask)!=n_doc:
                    q_relCur[mask]=-np.inf
            item_ij=np.argmax(q_relCur)
            QuotaEachItem[item_ij]-=positionBias[j]
            QuotaEachItem=np.clip(QuotaEachItem,0,np.inf)
            ranking.append(item_ij)
            Expo[item_ij]+=positionBias[j]
    return rankLists
def getVerticalRanking(q_rel,rankListLength,n_futureSession,QuotaEachItem,positionBias):
    """
    This funciton outputs ranklists by an vertical way.
    """
    Expo=np.zeros_like(q_rel)
    rankLists=[]
    n_doc=len(q_rel)
    for i in range(n_futureSession):
        rankLists.append([])
    for j in range(rankListLength):
        for i in range(n_futureSession):
            ranking=rankLists[i]
            q_relCur=q_rel+np.random.uniform(0,0.001,q_rel.shape) #to break tie
            q_relCur[ranking]=-np.inf
            # if QuotaEachItem.sum()>=ExpoBackwardCum[i,j]:
            QuotaSatisfiedId=np.where(QuotaEachItem<=positionBias[j]/2)[0]
            mask=list(set(QuotaSatisfiedId.tolist()+ranking))
            if len(mask)!=n_doc:
                q_relCur[mask]=-np.inf
            item_ij=np.argmax(q_relCur)
            QuotaEachItem[item_ij]-=positionBias[j]
            QuotaEachItem=np.clip(QuotaEachItem,0,np.inf)
            ranking.append(item_ij)
            Expo[item_ij]+=positionBias[j]
    return rankLists
def getHorizontalRanking(q_rel,rankListLength,n_futureSession,QuotaEachItem,positionBias):
    """
    This funciton outputs ranklists by an vertical way.
    """
    Expo=np.zeros_like(q_rel)
    rankLists=[]
    n_doc=len(q_rel)
    for i in range(n_futureSession):
        rankLists.append([])
    for i in range(n_futureSession):
      for j in range(rankListLength):
            ranking=rankLists[i]
            q_relCur=q_rel+np.random.uniform(0,0.001,q_rel.shape) #to break tie
            q_relCur[ranking]=-np.inf
            # if QuotaEachItem.sum()>=ExpoBackwardCum[i,j]:
            QuotaSatisfiedId=np.where(QuotaEachItem<=positionBias[j]/2)[0]
            mask=list(set(QuotaSatisfiedId.tolist()+ranking))
            if len(mask)!=n_doc:
                q_relCur[mask]=-np.inf
            item_ij=np.argmax(q_relCur)
            QuotaEachItem[item_ij]-=positionBias[j]
            QuotaEachItem=np.clip(QuotaEachItem,0,np.inf)
            ranking.append(item_ij)
            Expo[item_ij]+=positionBias[j]
    return rankLists
def getFutureRankingQuota(obs,q_rel,positionBias,n_futureSession,rankListLength,fairness_tradeoff_param):
    """
    This funciton outputs future rankists by conidering fair exposure gurantee.
    """    
    QuotaEachItem=getQuotaEachItemQuota(obs,q_rel,positionBias,n_futureSession,fairness_tradeoff_param)
    ExpoBackwardCum=getExpoBackwardCum(n_futureSession,rankListLength,positionBias)
    rankLists=getVerticalRankingPartial(q_rel,rankListLength,n_futureSession,ExpoBackwardCum,QuotaEachItem,positionBias)
    random.shuffle(rankLists)
    return rankLists
def getFutureRankingNDCG(obs,q_rel,positionBias,n_futureSession,rankListLength,fairness_tradeoff_param,exploration_tradeoff_param):
    """
    This funciton outputs future rankists vertically by conidering fair exposure gurantee.
    """    
    QuotaEachItem=getQuotaEachItemNDCG(obs,q_rel,positionBias,n_futureSession,fairness_tradeoff_param,exploration_tradeoff_param)
    # ExpoBackwardCum=getExpoBackwardCum(n_futureSession,rankListLength,positionBias)
    rankLists=getVerticalRanking(q_rel,rankListLength,n_futureSession,QuotaEachItem,positionBias)
    random.shuffle(rankLists)
    return rankLists
def getFutureRankingNDCGHorizontal(obs,q_rel,positionBias,n_futureSession,rankListLength,fairness_tradeoff_param,exploration_tradeoff_param):
    """
    This funciton outputs future rankists horizontally by conidering fair exposure gurantee.
    """    
    QuotaEachItem=getQuotaEachItemNDCG(obs,q_rel,positionBias,n_futureSession,fairness_tradeoff_param,exploration_tradeoff_param)
    # ExpoBackwardCum=getExpoBackwardCum(n_futureSession,rankListLength,positionBias)
    rankLists=getHorizontalRanking(q_rel,rankListLength,n_futureSession,QuotaEachItem,positionBias)
    random.shuffle(rankLists)
    return rankLists
def get_rankingFromDatasplit(qid,dataSplit,**kwargs):
    """
    This function return the ranking.
    """
    Rel=dataSplit.getEstimatedAverageRelevance()
    qRel=dataSplit.query_values_from_vector(qid,Rel)
    qExpVector=dataSplit.query_values_from_vector(qid,dataSplit.exposure)
    q_ItemFreqEachRank=dataSplit.query_values_from_vector(qid,dataSplit.ItemFreqEachRank)
    cacheLists=dataSplit.cacheLists[qid]
    kwargs["qid"]=qid
    kwargs["ItemFreqEachRank"]=q_ItemFreqEachRank
    ranking=get_ranking(qRel,qExpVector,cacheLists=cacheLists,dataSplit=dataSplit,\
      queryFreq=dataSplit.query_freq[qid],**kwargs)
    return ranking
def simClickForDatasplit(qid,dataSplit,ranking,positionBias,clickRNG,**kwargs):
    """
    This function return the ranking.
    """
    TrueRel=dataSplit.query_values_from_vector(qid,dataSplit.label_vector)
    clicks=sim.generateClick(ranking,TrueRel,positionBias,clickRNG)
    return clicks
def get_rankingFromData(data,userFeature,positionBias,**kwargs):
    """
    This function return the ranking.
    """
    qRel=data.getEstimatedAverageRelevance(userFeature)
    qExpVector=data.exposure
    cacheLists=data.cacheLists
    queryFreq=data.queryFreq
    ranking=get_ranking(qRel=qRel,qExpVector=qExpVector,positionBias=positionBias,cacheLists=cacheLists,\
      queryFreq=queryFreq,data=data,**kwargs)
    return ranking
def uncertaintyDoc(qExpVector,exploration_strategy):
    """
    This function return the marginal uncertainty.
    """  
    if exploration_strategy=="MarginalUncertainty":
      return 1/np.clip(qExpVector**2,0.1,np.inf)
    elif exploration_strategy==None:
      return np.zeros_like(qExpVector)
def MMF(ItemAverageRele,ItemFreqEachRank,positionBias,rankListLength,fairness_tradeoff_param=0.5,G=None,personal_relevance=None):
    """
    # This function return the ranklist by using MMF method. https://arxiv.org/abs/2102.09670
    """  
    NumItems=ItemFreqEachRank.shape[0]
    
    if G is None:  ## then we are going to do individual fairness.
        G=[[i]for i in range(NumItems)]
    NumG=len(G)
    itemRange=np.arange(NumItems)
    GroupRange=np.arange(NumG)
    ItemCumExpoEachRank=np.cumsum(ItemFreqEachRank*positionBias[np.newaxis,:],axis=1)
    GroupCumExpoEachRank=np.array([np.mean(ItemCumExpoEachRank[i],axis=0) for i in G])
    GroupAverageRele=np.array([np.mean(ItemAverageRele[i]) for i in G])
    GroupAverageRele=np.clip(GroupAverageRele,0.001,np.inf)
    G_size=np.array([len(G_i) for G_i in G])
    GroupCumExpoOverReleEachRank=GroupCumExpoEachRank/GroupAverageRele[:,None]/G_size[:,None]
    GroupCumExpoOverReleEachRank=GroupCumExpoOverReleEachRank+np.random.uniform(0,0.001,size=(NumG,rankListLength)) ## randomize when two scores tie
    if personal_relevance is None:
        personal_relevance=ItemAverageRele
    ranking=[]
    AvailableItemMask=np.ones(NumItems).astype(np.bool_)
    for Rank_i in range(rankListLength):
        AvailableItems=itemRange[AvailableItemMask]
        if random.random()>fairness_tradeoff_param:   ## select according to relevance.
            SelectedId=AvailableItems[np.argmax(ItemAverageRele[AvailableItemMask])]
        else:
            AvailableGroupMask=np.array([AvailableItemMask[i].sum()>0 for i in G]).astype(np.bool_)
            AvailableGroup=GroupRange[AvailableGroupMask]
            GroupId=AvailableGroup[np.argmin(GroupCumExpoOverReleEachRank[AvailableGroupMask,Rank_i])] ## find out the most underexposed group.
            FilteredId=G[GroupId]  ## select items in this underexposed groups
            AvailableItems=itemRange[FilteredId][AvailableItemMask[FilteredId]] ## select unmasked items in this underexposed groups
            SelectedId=AvailableItems[np.argmax(ItemAverageRele[AvailableItems])]
        AvailableItemMask[SelectedId]=np.False_
        ranking.append(SelectedId)
    ranking=np.array(ranking).astype(int)
    # unique, counts = np.unique(ranking, return_counts=True)
    # assert unique.shape[0]==ranking.shape[0],"there should not be repeated items"
    return ranking


def PLFairRanking(model,q_feat,q_cutoff=5):
    q_tf_scores = model(q_feat)

    q_np_scores = q_tf_scores.numpy()[:,0]

    sampled_rankings = pl.gumbel_sample_rankings(
                                    q_np_scores,
                                    1,
                                    cutoff=q_cutoff)[0][0]
    return sampled_rankings

    
def get_ranking(qRel,qExpVector,fairness_strategy,fairness_tradeoff_param,exploration_strategy,exploration_tradeoff_param,\
  rankListLength,n_futureSession=None,positionBias=None,cacheLists=[],queryFreq=0,dataSplit=None,**kwargs):

    num_item=qRel.shape[0]
    if fairness_strategy in ["FairCo",'FairCo_multip.',"GradFair","FairCo_maxnorm"]:
      Docunfairness=unfairnessDoc(qExpVector,qRel,fairness_strategy)
      Uncertainty=uncertaintyDoc(qExpVector,exploration_strategy)
      RankingScore=qRel+fairness_tradeoff_param*Docunfairness+exploration_tradeoff_param*Uncertainty
      ranking=single_ranking(RankingScore,rankListLength=rankListLength)
    elif fairness_strategy in ["onlyFairness"]:
      Docunfairness=unfairnessDoc(qExpVector,qRel,"GradFair")
      Uncertainty=uncertaintyDoc(qExpVector,exploration_strategy)
      RankingScore=Docunfairness+exploration_tradeoff_param*Uncertainty
      ranking=single_ranking(RankingScore,rankListLength=rankListLength)
    # elif fairness_strategy in ["QPfair"]:
    #   if len(cacheLists)<=0:
    #     cacheLists+=getFutureRankingQuota(qExpVector,qRel,positionBias,n_futureSession,rankListLength,fairness_tradeoff_param)
    #   ranking=np.array(cacheLists.pop())
    elif fairness_strategy in ["QPFair"]:
      if len(cacheLists)<=0:
        cacheLists+=getFutureRankingNDCG(qExpVector,qRel,positionBias,n_futureSession,rankListLength,fairness_tradeoff_param,exploration_tradeoff_param)
      ranking=np.array(cacheLists.pop())
    elif fairness_strategy in ["QPFair-Horiz."]:
      if len(cacheLists)<=0:
        cacheLists+=getFutureRankingNDCGHorizontal(qExpVector,qRel,positionBias,n_futureSession,rankListLength,fairness_tradeoff_param,exploration_tradeoff_param)
      ranking=np.array(cacheLists.pop())
    elif fairness_strategy == "Randomk":
      RankingScore=np.random.uniform(0,1,qRel.shape)
      ranking=single_ranking(RankingScore,rankListLength=rankListLength)
    elif fairness_strategy == "MMF":
      RankingScore=qRel
      ItemFreqEachRank=kwargs["ItemFreqEachRank"]
      ranking=MMF(RankingScore,ItemFreqEachRank,positionBias,rankListLength=rankListLength,fairness_tradeoff_param=fairness_tradeoff_param)
    elif fairness_strategy == "FairK":
      Docunfairness=unfairnessDoc(qExpVector,qRel,"GradFair")
      RankingScore=Docunfairness
      ranking=single_ranking(RankingScore,rankListLength=rankListLength)  
    elif fairness_strategy == "ExploreK":
      Uncertainty=uncertaintyDoc(qExpVector,exploration_strategy)
      RankingScore=Uncertainty
      ranking=single_ranking(RankingScore,rankListLength=rankListLength)    
    elif fairness_strategy == "Topk":
      RankingScore=qRel
      ranking=single_ranking(RankingScore,rankListLength=rankListLength)
    elif fairness_strategy == "PLFair":
      if dataSplit.PLFairModel is None or dataSplit.query_freq.sum()%n_futureSession==0:
        dataSplit.PLFairModel=PLFairTrain.TrainPLFairModel(dataSplit,positionBias,rankListLength=5,fairness_tradeoff_param=fairness_tradeoff_param)
      qid=kwargs["qid"]
      q_feat = dataSplit.query_feat(qid)
      ranking=PLFairRanking(dataSplit.PLFairModel,q_feat,q_cutoff=rankListLength)
    elif fairness_strategy == "ILP":
      ranking=ILP(positionBias,qRel,qExpVector,queryFreq,rankListLength,fairness_tradeoff_param)
    elif fairness_strategy == "LP":
      positionBiasAllItem=np.zeros(num_item)
      positionBiasAllItem[:rankListLength]=positionBias
      qidDecomps=dataSplit.decomps[qid]
      if queryFreq%n_futureSession==0:
        decomp=LP(positionBiasAllItem,qRel,ind_fair=True, group_fair=False, debug=False,\
          w_fair =fairness_tradeoff_param, group_click_rel = None, impact=False, LP_COMPENSATE_W=None)
        qidDecomps.decomp=decomp
      ranking=sampleFromLp(qidDecomps.decomp)
      ranking=ranking[:rankListLength]
    return ranking
