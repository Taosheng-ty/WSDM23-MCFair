import sys
import json
import os
scriptPath=os.path.dirname(os.path.abspath(__file__))
os.chdir(scriptPath+"/../..")
import BatchExpLaunch.tools as tools
root_path="localOutput/MQ2008/"
def write_setting(datasets,list_settings,settings_base,shown_prob=None):
    """
    This function write settings.json which specify the parameters of main function.
    """
    
    for dataset in datasets:
        list_settings_data=dict(list_settings)
        list_settings_data["dataset_name"]=[dataset]
        list_settings_data = {k: list_settings_data[k] for k in desired_order_list if k in list_settings_data}
        setting_data=dict(settings_base)
        setting_data={**setting_data, **dataset_dict[dataset]} 
        print(root_path,"x"*100)
        tools.iterate_settings(list_settings_data,setting_data,path=root_path) 


settings_base={
        "progressbar":"false",
        "rankListLength":5,
        "query_least_size":5,
        # "NumDocMaximum":20,
        # "relvance_strategy":"EstimatedAverage"
        }
# root_path="localOutput/Feb182022Data/"
positionBiasSeverity=[1]

# root_path="localOutput/MCFairAug02-2022LTR/"
desired_order_list=["relvance_strategy",'positionBiasSeverity',"dataset_name","fairness_strategy","n_futureSession","fairness_tradeoff_param","exploration_tradeoff_param","random_seed"]


##############################post-processing
datasets=["MQ2008"]
dataset_dict={"MQ2008":{"n_iteration":10**4,"queryMaximumLength":20}}

list_settings={"relvance_strategy":["TrueAverage"],'positionBiasSeverity':positionBiasSeverity,"fairness_strategy":['PLFair'],"fairness_tradeoff_param":[0.0,0.01,0.05,0.1,0.2,0.5,0.8,0.9,1.0],\
              "exploration_tradeoff_param":[0.0], "random_seed":[0,1,2,3,4],"n_futureSession":[10000000]}
write_setting(datasets,list_settings,settings_base)
##write setting.json for Topk and Randomk
list_settings={"relvance_strategy":["TrueAverage"],'positionBiasSeverity':positionBiasSeverity,"fairness_strategy":['MMF'],"fairness_tradeoff_param":[0.0,0.01,0.05,0.1,0.2,0.5,0.8,0.9,1.0],\
              "exploration_tradeoff_param":[0.0], "random_seed":[0,1,2,3,4]}
write_setting(datasets,list_settings,settings_base)




##write setting.json for 'FairCo', 'FairCo_multip.','FairCo_average'
list_settings={"relvance_strategy":["TrueAverage"],'positionBiasSeverity':positionBiasSeverity,"fairness_strategy":['FairCo', 'GradFair'],"fairness_tradeoff_param":[0.0,0.0001,0.001,0.005,0.01,0.1,0.5,1,10,50,100,500,700,1000],\
              "exploration_tradeoff_param":[0.0], "random_seed":[0,1,2,3,4]}
write_setting(datasets,list_settings,settings_base)


##write setting.json for Topk and Randomk

list_settings={"relvance_strategy":["TrueAverage"],'positionBiasSeverity':positionBiasSeverity,"fairness_strategy":['Topk','Randomk',"FairK","ExploreK"],"random_seed":[0,1,2,3,4]}
write_setting(datasets,list_settings,settings_base)

##write setting.json for ILP and LP  only for MQ2008
datasets=["MQ2008"]
list_settings={"relvance_strategy":["TrueAverage"],'positionBiasSeverity':positionBiasSeverity,"fairness_strategy":[ 'ILP'],"fairness_tradeoff_param":[0.0,0.01,0.05,0.1,0.2,0.5,0.8,0.9,1.0],\
               "exploration_tradeoff_param":[0.0],"random_seed":[0,1,2,3,4]}
write_setting(datasets,list_settings,settings_base)
list_settings={"relvance_strategy":["TrueAverage"],'positionBiasSeverity':positionBiasSeverity,"fairness_strategy":[ 'LP'],"fairness_tradeoff_param":[0.0,0.1,1.0,2,10,50,100,500,1000],\
               "exploration_tradeoff_param":[0.0],"random_seed":[0,1,2,3,4],"n_futureSession":[100000]}
# settings_base_LP=dict(settings_base)
# settings_base_LP["n_futureSession"]=200
write_setting(datasets,list_settings,settings_base) 








##############################in-processing
datasets=["MQ2008"]
dataset_dict={"MQ2008":{"n_iteration":10**5,"queryMaximumLength":20}}
##write setting.json for 'FairCo', 'FairCo_multip.','FairCo_average'
list_settings={"relvance_strategy":["EstimatedAverage"],'positionBiasSeverity':positionBiasSeverity,"fairness_strategy":['PLFair'],"fairness_tradeoff_param":[0.0,0.01,0.05,0.1,0.2,0.5,0.8,0.9,1.0],\
              "exploration_tradeoff_param":[0.0], "random_seed":[0,1,2,3,4],"n_futureSession":[int(dataset_dict["MQ2008"]["n_iteration"]/10)]}
write_setting(datasets,list_settings,settings_base)
list_settings={"relvance_strategy":["EstimatedAverage"],'positionBiasSeverity':positionBiasSeverity,"fairness_strategy":['MMF'],"fairness_tradeoff_param":[0.0,0.01,0.05,0.1,0.2,0.5,0.8,0.9,1.0],\
              "exploration_tradeoff_param":[0.0], "random_seed":[0,1,2,3,4]}
write_setting(datasets,list_settings,settings_base)


list_settings={"relvance_strategy":["EstimatedAverage"],'positionBiasSeverity':positionBiasSeverity,"fairness_strategy":[ 'GradFair'],"fairness_tradeoff_param":[0.0,0.1,0.4,0.5,0.6,0.7,0.8,0.9,1,2,5,10,20,50,80,100,200,250,300,400,500,600,620,640,645,650,680,685,690,700,1000],\
        #        "exploration_tradeoff_param":[0.0,0.1,0.5,1,5,10,20,50,100],
               "exploration_tradeoff_param":[50,100],
               "random_seed":[0,1,2,3,4]}
write_setting(datasets,list_settings,settings_base)

list_settings={"relvance_strategy":["EstimatedAverage"],'positionBiasSeverity':positionBiasSeverity,"fairness_strategy":['FairCo'],"fairness_tradeoff_param":[0.0,0.00001,0.0001,0.0005,0.001,0.005,0.01,0.1,0.5,1,10,50,100,500,700,1000],\
               "exploration_tradeoff_param":[0.0,0.1,1,10,20,50,100],"random_seed":[0,1,2,3,4]}
write_setting(datasets,list_settings,settings_base)

list_settings={"relvance_strategy":["EstimatedAverage"],'positionBiasSeverity':positionBiasSeverity,"fairness_strategy":["onlyFairness"],"fairness_tradeoff_param":[1],\
               "exploration_tradeoff_param":[0.0,0.1,0.5,1,5,10,20,100,200,1000],"random_seed":[0,1,2,3,4]}
write_setting(datasets,list_settings,settings_base)



##write setting.json for Topk and Randomk

list_settings={"relvance_strategy":["EstimatedAverage"],'positionBiasSeverity':positionBiasSeverity,"fairness_strategy":['Topk','Randomk',"FairK","ExploreK"],"random_seed":[0,1,2,3,4]}
write_setting(datasets,list_settings,settings_base)

##write setting.json for ILP and LP  only for MQ2008
list_settings={"relvance_strategy":["EstimatedAverage"],'positionBiasSeverity':positionBiasSeverity,"fairness_strategy":[ 'ILP'],"fairness_tradeoff_param":[0.0,0.01,0.1,0.2,0.5,0.8,0.9,1.0],\
               "exploration_tradeoff_param":[0.0],"random_seed":[0,1,2,3,4]}
write_setting(datasets,list_settings,settings_base) 

list_settings={"relvance_strategy":["EstimatedAverage"],'positionBiasSeverity':positionBiasSeverity,"fairness_strategy":[ 'LP'],"fairness_tradeoff_param":[0.0,0.01,0.05,0.1,0.2,0.5,1.0,2,10,50,80,100,500,1000],\
               "exploration_tradeoff_param":[0.0],"random_seed":[0,1,2,3,4],"n_futureSession":[100]}
# settings_base_LP=dict(settings_base)
# settings_base_LP["n_futureSession"]=200
write_setting(datasets,list_settings,settings_base) 


##write setting.json for 'QPfair',"Hybrid","QPfairNDCG"

# list_settings={"relvance_strategy":["TrueAverage"],'positionBiasSeverity':[0,1,2],"fairness_strategy":["QPFair","QPFair-Horiz."],"fairness_tradeoff_param":[0.0,0.01,0.05,0.1,0.2,0.5,0.8,0.85,0.9,0.92,0.95,0.98,1.0],\
#                "random_seed":[0,1,2,3,4],"n_futureSession":[2,5,10,20,50,100,200,500]}
# write_setting(datasets,list_settings,settings_base)

##write setting.json for 'QPfair',"Hybrid","QPfairNDCG" in realworld setting

# list_settings={"relvance_strategy":["EstimatedAverage"],'positionBiasSeverity':[0,1,2],"fairness_strategy":["QPFair","QPFair-Horiz."],"fairness_tradeoff_param":[0.0,0.01,0.05,0.1,0.2,0.5,0.8,0.85,0.9,0.92,0.95,0.98,1.0],\
#               "exploration_tradeoff_param":[0,3,5,10,20], "random_seed":[0,1,2,3,4],"n_futureSession":[10,20,50,100,200,500]}
# write_setting(datasets,list_settings,settings_base)


