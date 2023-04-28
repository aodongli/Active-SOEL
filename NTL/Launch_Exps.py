
import argparse
import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
from config.base import Grid, Config
from evaluation.Experiments import runExperiment
from evaluation.Kvariants_Eval import KVariantEval

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config-file', dest='config_file', default='config_img_active.yml')
    parser.add_argument('--dataset-name', dest='dataset_name', default='blood')
    parser.add_argument('--contamination', type=float, default=0.1)
    parser.add_argument('--query_num', type=int, default=20)
    return parser.parse_args()

def EndtoEnd_Experiments(config_file, dataset_name,contamination,query_num):

    model_configurations = Grid(config_file, dataset_name)
    model_configuration = Config(**model_configurations[0])
    dataset =model_configuration.dataset
    result_folder = model_configuration.result_folder+model_configuration.exp_name
    exp_path = os.path.join(result_folder,f'{contamination}_{model_configuration.train_method}_{model_configuration.query_method}_{query_num}')

    risk_assesser = KVariantEval(dataset, exp_path, model_configurations,contamination,query_num)
    risk_assesser.risk_assessment(runExperiment)

if __name__ == "__main__":
    args = get_args()
    config_file = 'config_files/'+args.config_file
    EndtoEnd_Experiments(config_file, args.dataset_name,args.contamination,args.query_num)
