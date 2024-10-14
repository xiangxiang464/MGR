import os
import argparse
import torch
import torch.nn as nn
from modeling import models
from utils import config_loader, get_ddp_module, init_seeds, params_count, get_msg_mgr

parser = argparse.ArgumentParser(description='Main program for opengait.')
parser.add_argument('--world_size', default=1, type=int,
                    help='number of distributed processes')
# parser.add_argument("--local_rank", type=int,
#                     help='local rank for DistributedDataParallel')
parser.add_argument('--local_rank', type=int, default=0,
                    help="passed by torch.distributed.launch module")
parser.add_argument('--cfgs', type=str,
                    default='config/default.yaml', help="path of config file")
parser.add_argument('--phase', default='train',
                    choices=['train', 'test'], help="choose train or test phase")
parser.add_argument('--log_to_file', action='store_true',
                    help="log to file, default path is: output/<dataset>/<model>/<save_name>/<logs>/<Datetime>.txt")
parser.add_argument('--iter', default=0, help="iter to restore")
opt = parser.parse_args()

def initialization(cfgs, training):
    msg_mgr = get_msg_mgr()
    engine_cfg = cfgs['trainer_cfg'] if training else cfgs['evaluator_cfg']
    output_path = os.path.join('output/', cfgs['data_cfg']['dataset_name'],
                               cfgs['model_cfg']['model'], engine_cfg['save_name'])
    if training:
        msg_mgr.init_manager(output_path, opt.log_to_file, engine_cfg['log_iter'],
                             engine_cfg['restore_hint'] if isinstance(engine_cfg['restore_hint'], (int)) else 0)
    else:
        msg_mgr.init_logger(output_path, opt.log_to_file)

    msg_mgr.log_info(engine_cfg)

    # 移除根据 rank 初始化种子的逻辑，直接使用固定种子
    seed = 42
    init_seeds(seed)

def run_multiple_models(cfgs_list, training):
    msg_mgr = get_msg_mgr()

    # 遍历配置文件列表，每个配置文件生成一个模型
    for i, cfgs in enumerate(cfgs_list):
        model_cfg = cfgs['model_cfg']
        msg_mgr.log_info(f"Building model {i}: {model_cfg}")

        Model = getattr(models, model_cfg['model'])
        model = Model(cfgs, training)

        # if training and cfgs['trainer_cfg']['sync_BN']:
        #     model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        if cfgs['trainer_cfg']['fix_BN']:
            model.fix_BN()

        # model = get_ddp_module(model)
        msg_mgr.log_info(params_count(model))
        msg_mgr.log_info(f"Model {i} Initialization Finished!")

        # 训练或测试每个模型
        if training:
            Model.run_train(model)
        else:
            Model.run_fused_test(model)


if __name__ == '__main__':
    # init_distributed_mode(opt)

    # torch.distributed.init_process_group('nccl', init_method='env://')
    # if torch.distributed.get_world_size() != torch.cuda.device_count():
    #     raise ValueError("Expect number of availuable GPUs({}) equals to the world size({}).".format(
    #         torch.cuda.device_count(), torch.distributed.get_world_size()))

    # torch.distributed.init_process_group('nccl', init_method='env://',
    #                                      world_size=opt.world_size, rank=opt.rank)
    # # 多机多卡
    # if torch.distributed.get_world_size() > 1:
    #     tensor_list = []
    #     for dev_idx in range(torch.cuda.device_count()):
    #         tensor_list.append(torch.FloatTensor([1]).cuda(dev_idx))
    #     torch.distributed.all_reduce_multigpu(tensor_list)
    # # 单机多卡
    # else:
    #     if torch.distributed.get_world_size() != torch.cuda.device_count():
    #         raise ValueError("Expect number of availuable GPUs({}) equals to the world size({}).".format(
    #             torch.cuda.device_count(), torch.distributed.get_world_size()))

    cfgs = config_loader(opt.cfgs)
    # if opt.iter != 0:
    #     cfgs['evaluator_cfg']['restore_hint'] = int(opt.iter)
    #     cfgs['trainer_cfg']['restore_hint'] = int(opt.iter)

    training = (opt.phase == 'train')
    initialization(cfgs, training)
    cfgs_list = [
        config_loader('configs/fused/model1.yaml'),
        config_loader('configs/fused/model2.yaml'),
        config_loader('configs/fused/model3.yaml'),
    ]
    run_multiple_models(cfgs_list, training)
