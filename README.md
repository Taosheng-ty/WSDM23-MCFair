# WSDM22-MCFair
WSDM 2022 paper. Marginal-Certainty-aware Fair Ranking Algorithm. 
The paper could be found at https://arxiv.org/pdf/2212.09031.pdf


## Create the env.
    conda env create -f environment.yml
then activate the env 

    conda activate MCFair
## Specify the data directory 
We specify data directory in LTRlocal_dataset_info.txt.

## Then you can run the default setting 
    python main.py
and specify the arguments accordingly.

## How to launch experiments of different settings?
we provide scripts to run multiple experiments. Set MQ2008 as an example, firstly, we run the following to generate the json settings. 

    python scripts/datascriptsGradFairLTR/generatingSettingMQ2008.py

Then, you can submit the whole MQ2008 experiments 

    slurm_python --CODE_PATH=.  --Cmd_file=main.py --JSON_PATH=localOutput/MQ2008  --jobs_limit=10  --secs_each_sub=5 --json2args  --plain_scripts  --only_unfinished
Here slurm_python is a tool to submit multiple jobs and it will be automatically installed when you create the conda env. You can change  jobs_limit to set how many jobs to run at the same time. If you have slurm installed in you server, you can remove --plain_scripts to let slurm schedule the jobs.

if you use MCFair in your research, please use the following BibTex entry.

    @article{yang2022marginal,
    title={Marginal-Certainty-aware Fair Ranking Algorithm},
    author={Yang, Tao and Xu, Zhichao and Wang, Zhenduo and Tran, Anh and Ai, Qingyao},
    journal={arXiv preprint arXiv:2212.09031},
    year={2022}
    }







