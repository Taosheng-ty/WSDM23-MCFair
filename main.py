import numpy as np
import utils.dataset as dataset
import utils.simulation as sim
import utils.ranking as rnk
import utils.evaluation as evl
from collections import defaultdict
from progressbar import progressbar
import argparse
from str2bool import str2bool
import json
import os
import random
import sys
import time
from BatchExpLaunch import  results_org
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", type=str,
                        default="localOutput/",
                        help="Path to result logs")
    parser.add_argument("--dataset_name", type=str,
                        default="MQ2008",
                        help="Name of dataset to sample from.")
    parser.add_argument("--dataset_info_path", type=str,
                        default="LTRlocal_dataset_info.txt",
                        help="Path to dataset info file.")
    parser.add_argument("--fold_id", type=int,
                        help="Fold number to select, modulo operator is applied to stay in range.",
                        default=1)
    parser.add_argument("--query_least_size", type=int,
                        default=5,
                        help="query_least_size, filter out queries with number of docs less than this number.")
    parser.add_argument("--queryMaximumLength", type=int,
                    default=np.inf,
                    help="the Maximum number of docs")
    parser.add_argument("--rankListLength", type=int,
                    help="Maximum number of items that can be displayed.",
                    default=5)
    parser.add_argument("--fairness_strategy", type=str,
                        choices=['FairCo', 'FairCo_multip.',"onlyFairness", 'GradFair',"Randomk","FairK",\
                            "ExploreK","Topk","FairCo_maxnorm","QPFair","QPFair-Horiz.","ILP","LP","MMF","PLFair"],
                        default="FairCo",
                        help="fairness_strategy, available choice is ['FairCo', 'FairCo_multip.', 'QPFair','GradFair','Randomk','Topk']")
    parser.add_argument("--fairness_tradeoff_param", type=float,
                            default=0.5,
                            help="fairness_tradeoff_param")
    parser.add_argument("--relvance_strategy", type=str,
                        choices=['TrueAverage',"NNmodel","EstimatedAverage"],
                        default="TrueAverage",
                        help="relvance_strategy, available choice is ['TrueAverage', 'NNmodel.', 'EstimatedAverage']")
    parser.add_argument("--exploration_strategy", type=str,
                        choices=['MarginalUncertainty',None],
                        default='MarginalUncertainty',
                        help="exploration_strategy, available choice is ['MarginalUncertainty', None]")
    parser.add_argument("--exploration_tradeoff_param", type=float,
                            default=5,
                            help="exploration_tradeoff_param")
    parser.add_argument("--random_seed", type=int,
                    default=0,
                    help="random seed for reproduction")
    parser.add_argument("--positionBiasSeverity", type=int,
                    help="Severity of positional bias",
                    default=1)
    parser.add_argument("--n_iteration", type=int,
                    default=4*10**4,
                    help="how many iteractions to simulate")
    parser.add_argument("--n_futureSession", type=int,
                    default=1000000000,
                    help="how many future session we want consider in advance, only works if we use QPFair strategy.")
    parser.add_argument("--progressbar",  type=str2bool, nargs='?',
                    const=True, default=True,
                    help="use progressbar or not.")
    parser.add_argument("--LogTimeEachStep",  type=str2bool, nargs='?',
                const=False, default=False,
                help="use progressbar or not.")
    args = parser.parse_args()
    # args = parser.parse_args(args=[]) # for debug
    # load the data and filter out queries with number of documents less than query_least_size.
    argsDict=vars(args)
    voidFeature=False if args.fairness_strategy=="PLFair" else True
    data = dataset.get_data(args.dataset_name,
                  args.dataset_info_path,
                  args.fold_id,
                  args.query_least_size,
                  args.queryMaximumLength,
                  relvance_strategy=args.relvance_strategy,\
                  rankListLength= args.rankListLength,
                  voidFeature=voidFeature)
    # begin simulation
    Logging=results_org.getLogging()
    positionBias=sim.getpositionBias(args.rankListLength,args.positionBiasSeverity) 
    argsDict["positionBias"]=positionBias
    NDCGcutoffs=[i for i in [1,2,3,4,5,10,20] if i<=args.rankListLength]
    assert args.rankListLength>=args.query_least_size, print("For simplicity, the ranked list length should be greater than doc length")
    queryRndSeed=np.random.default_rng(args.random_seed) 
    random.seed(args.random_seed)
    clickRNG=np.random.default_rng(args.random_seed) 
    OutputDict=defaultdict(list)
    NDCGDict=defaultdict(list)
    evalIterations=np.linspace(0, args.n_iteration-1, num=21,endpoint=True).astype(np.int32)[1:]
    iterationsGenerator=progressbar(range(args.n_iteration)) if args.progressbar else range(args.n_iteration)
    start_time = time.time()
    n_testCounter=0
    timeLog=evl.TimeExecute()
    for iteration in iterationsGenerator:
        # sample data split and a query
        qid,dataSplit=sim.sample_queryFromdata(data,queryRndSeed,only_test=True)
        if iteration in evalIterations:
            Logging.info("current iteration"+str(iteration))
            OutputDict["iterations"].append(iteration)
            OutputDict["time"].append(time.time()-start_time)
            OutputDict["testIter"].append(n_testCounter)
            evl.outputNDCG(NDCGDict,OutputDict)
            evl.outputFairnessDatasplit(data,OutputDict)
        if dataSplit.name !="test":
            continue 
        if args.LogTimeEachStep:
            print(timeLog.getTime(),flush=True)
        n_testCounter+=1
        # get a ranking according to fairness strategy
        ranking=rnk.get_rankingFromDatasplit(qid,\
                                dataSplit,\
                                **argsDict)
        # update exposure statistics according to ranking
        clicks=rnk.simClickForDatasplit(qid,dataSplit,ranking,positionBias,clickRNG)        
        dataSplit.updateStatistics(qid,clicks,ranking,positionBias)
        # calculate metrics ndcg and unfairness.
        evl.Update_NDCG_multipleCutoffsDatasplit(ranking,qid,dataSplit,positionBias,NDCGcutoffs,NDCGDict)

    #write the results.
    os.makedirs(args.log_dir,exist_ok=True)
    logPath=args.log_dir+"/result.jjson"
    print('Writing results to %s' % OutputDict)
    with open(logPath, 'w') as f:
        json.dump(OutputDict, f)
        