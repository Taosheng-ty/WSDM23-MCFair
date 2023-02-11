import sys
import json
import os
scriptPath=os.path.dirname(os.path.abspath(__file__))
os.chdir(scriptPath+"/../..")
import BatchExpLaunch.tools as tools
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
root_path="localOutput/GradFairMay5LTR/"
root_path="localOutput/istella-s/"
desired_order_list=["relvance_strategy",'positionBiasSeverity',"dataset_name","fairness_strategy","n_futureSession","fairness_tradeoff_param","exploration_tradeoff_param","random_seed"]

#################### for post-processing
##write setting.json for 'FairCo', 'FairCo_multip.','FairCo_average'
# datasets=["MSLR-WEB10k"]
datasets=["istella-s"]
dataset_dict={"istella-s":{"n_iteration":10**6,"queryMaximumLength":int(1e10)},\
        "MSLR-WEB10k":{"n_iteration":4*10**5,"queryMaximumLength":int(1e10)}}


list_settings={"relvance_strategy":["TrueAverage"],'positionBiasSeverity':positionBiasSeverity,"fairness_strategy":['PLFair'],"fairness_tradeoff_param":[0.0,0.01,0.05,0.1,0.2,0.5,0.8,0.9,1.0],\
              "exploration_tradeoff_param":[0.0], "random_seed":[0,1,2,3,4],"n_futureSession":[10000000]}
write_setting(datasets,list_settings,settings_base)
##write setting.json for Topk and Randomk
list_settings={"relvance_strategy":["TrueAverage"],'positionBiasSeverity':positionBiasSeverity,"fairness_strategy":['MMF'],"fairness_tradeoff_param":[0.0,0.01,0.05,0.1,0.2,0.5,0.8,0.9,1.0],\
              "exploration_tradeoff_param":[0.0], "random_seed":[0,1,2,3,4]}
write_setting(datasets,list_settings,settings_base)


list_settings={"relvance_strategy":["TrueAverage"],'positionBiasSeverity':positionBiasSeverity,"fairness_strategy":['FairCo', 'GradFair'],"fairness_tradeoff_param":[0.0,0.0001,0.001,0.005,0.01,0.05,0.1,0.2,0.5,1,5,10,50,100,500,700,1000],\
              "exploration_tradeoff_param":[0.0], "random_seed":[0,1,2,3,4]}
write_setting(datasets,list_settings,settings_base)

##write setting.json for Topk and Randomk

list_settings={"relvance_strategy":["TrueAverage"],'positionBiasSeverity':positionBiasSeverity,"fairness_strategy":['Topk','Randomk',"FairK","ExploreK"],"random_seed":[0,1,2,3,4]}
write_setting(datasets,list_settings,settings_base)



#################### for in-processing

# datasets=["MSLR-WEB10k"]
datasets=["istella-s"]
dataset_dict={"istella-s":{"n_iteration":10**7,"queryMaximumLength":int(1e10)},\
        "MSLR-WEB10k":{"n_iteration":4*10**6,"queryMaximumLength":int(1e10)}}

list_settings={"relvance_strategy":["EstimatedAverage"],'positionBiasSeverity':positionBiasSeverity,"fairness_strategy":['PLFair'],"fairness_tradeoff_param":[0.0,0.01,0.05,0.1,0.2,0.5,0.8,0.9,1.0],\
              "exploration_tradeoff_param":[0.0], "random_seed":[0,1,2,3,4],"n_futureSession":[int(dataset_dict["istella-s"]["n_iteration"]/10)]}
write_setting(datasets,list_settings,settings_base)
list_settings={"relvance_strategy":["EstimatedAverage"],'positionBiasSeverity':positionBiasSeverity,"fairness_strategy":['MMF'],"fairness_tradeoff_param":[0.0,0.01,0.05,0.1,0.2,0.5,0.8,0.9,1.0],\
              "exploration_tradeoff_param":[0.0], "random_seed":[0,1,2,3,4]}
write_setting(datasets,list_settings,settings_base)

list_settings={"relvance_strategy":["EstimatedAverage"],'positionBiasSeverity':positionBiasSeverity,"fairness_strategy":[ 'GradFair'],"fairness_tradeoff_param":[0.0,0.0001,0.001,0.005,0.01,0.1,0.5,1,10,50,100,500,700,1000],\
               "exploration_tradeoff_param":[0.0,1,10,100],"random_seed":[0,1,2,3,4]}
write_setting(datasets,list_settings,settings_base)

list_settings={"relvance_strategy":["EstimatedAverage"],'positionBiasSeverity':positionBiasSeverity,"fairness_strategy":[ 'GradFair'],"fairness_tradeoff_param":[0.0],\
               "exploration_tradeoff_param":[0.0,0.1,0.5,1,5,10,100,1000],"random_seed":[0,1,2,3,4]}
write_setting(datasets,list_settings,settings_base)

list_settings={"relvance_strategy":["EstimatedAverage"],'positionBiasSeverity':positionBiasSeverity,"fairness_strategy":['FairCo'],"fairness_tradeoff_param":[0.0,0.0001,0.001,0.005,0.01,0.1,0.5,1,10,50,100,500,700,1000],\
               "exploration_tradeoff_param":[0.0,1,10,100],"random_seed":[0,1,2,3,4]}
write_setting(datasets,list_settings,settings_base)

list_settings={"relvance_strategy":["EstimatedAverage"],'positionBiasSeverity':positionBiasSeverity,"fairness_strategy":["onlyFairness"],"fairness_tradeoff_param":[1],\
               "exploration_tradeoff_param":[0.0,0.1,0.5,1,5,10,100,1000],"random_seed":[0,1,2,3,4]}
write_setting(datasets,list_settings,settings_base)

##write setting.json for Topk and Randomk

list_settings={"relvance_strategy":["EstimatedAverage"],'positionBiasSeverity':positionBiasSeverity,"fairness_strategy":['Topk','Randomk',"FairK","ExploreK"],"random_seed":[0,1,2,3,4]}
write_setting(datasets,list_settings,settings_base)







# settings_base_LP=dict(settings_base)
# settings_base_LP["n_futureSession"]=200
# write_setting(datasets,list_settings,settings_base) 

##write setting.json for 'QPfair',"Hybrid","QPfairNDCG"

# list_settings={"relvance_strategy":["TrueAverage"],'positionBiasSeverity':[0,1,2],"fairness_strategy":["QPFair","QPFair-Horiz."],"fairness_tradeoff_param":[0.0,0.01,0.05,0.1,0.2,0.5,0.8,0.85,0.9,0.92,0.95,0.98,1.0],\
#                "random_seed":[0,1,2,3,4],"n_futureSession":[2,5,10,20,50,100,200,500]}
# write_setting(datasets,list_settings,settings_base)

##write setting.json for 'QPfair',"Hybrid","QPfairNDCG" in realworld setting

# list_settings={"relvance_strategy":["EstimatedAverage"],'positionBiasSeverity':[0,1,2],"fairness_strategy":["QPFair","QPFair-Horiz."],"fairness_tradeoff_param":[0.0,0.01,0.05,0.1,0.2,0.5,0.8,0.85,0.9,0.92,0.95,0.98,1.0],\
#               "exploration_tradeoff_param":[0,3,5,10,20], "random_seed":[0,1,2,3,4],"n_futureSession":[10,20,50,100,200,500]}
# write_setting(datasets,list_settings,settings_base)


