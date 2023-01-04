import argparse
from pcg import NCL
from trainer import NCLTrainer
from logging import getLogger
from recbole.trainer import HyperTuning
import logging
from recbole.config import Config
from recbole.data import create_dataset, data_preparation, save_split_dataloaders, load_split_dataloaders
from recbole.utils import init_logger, get_model, get_trainer, init_seed, set_color

# 'jester', 'ml-1m', 'yelp'
DS = "ml-1m"

def objective_function(config_dict=None, config_file_list=None, saved=True):
    r""" The default objective_function used in HyperTuning

    Args:
        config_dict (dict, optional): Parameters dictionary used to modify experiment parameters. Defaults to ``None``.
        config_file_list (list, optional): Config files used to modify experiment parameters. Defaults to ``None``.
        saved (bool, optional): Whether to save the model. Defaults to ``True``.
    """

    # 基于传入的设置参数创建设置变量
    config = Config(model=NCL, dataset=DS, config_dict=config_dict, config_file_list=config_file_list)
    logging.basicConfig(level=logging.ERROR)

    # 设定种子确保可以复现
    init_seed(config['seed'], config['reproducibility'])

    # 创建和切分数据集
    dataset = create_dataset(config)
    train_data, valid_data, test_data = data_preparation(config, dataset)
    # model = get_model(config['model'])(config, train_data.dataset).to(config['device'])

    # 创建模型，加载到显卡
    model = NCL(config, train_data.dataset).to(config['device'])
    # trainer = get_trainer(config['MODEL_TYPE'], config['model'])(config, model)

    # 创建训练器，开始训练和最终测试
    trainer = NCLTrainer(config, model)
    best_valid_score, best_valid_result = trainer.fit(
                    train_data, valid_data, saved=saved, show_progress=config['show_progress'])
    test_result = trainer.evaluate(test_data, load_best_model=saved)

    # 返回结果
    return {
        'best_valid_score': best_valid_score,
        'valid_score_bigger': config['valid_metric_bigger'],
        'best_valid_result': best_valid_result,
        'test_result': test_result
    }


def parameter_tuning():
    # 准备训练参数设置
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_files', type=str, default=None, help='fixed config files')
    parser.add_argument('--params_file', type=str, default='parameter_tuning/model.hyper', help='parameters file')
    parser.add_argument('--output_file', type=str, default='parameter_tuning/hyper_example.result', help='output file')

    parser.add_argument('--dataset', type=str, default=DS,
                        help='The datasets can be: jester, ml-1m, yelp, amazon-books, gowalla-merged, alibaba.')
    parser.add_argument('--config', type=str, default='', help='External config file name.')
    args, _ = parser.parse_known_args()

    # 指定通用和模型设置文档
    args.config_file_list = [
        'properties/overall.yaml',
        'properties/PCG.yaml',
    ]

    # 指定数据集设置文档
    if args.dataset in ['jester', 'ml-1m', 'yelp', 'amazon-books', 'gowalla-merged', 'alibaba']:
        args.config_file_list.append(f'properties/{args.dataset}.yaml')
    if args.config != '':
        args.config_file_list.append(args.config)

    # 提取设置文件列表
    config_file_list = args.config_file_list

    # 构建设置变量，用于logger日志记录初始化
    config = Config(model=NCL, dataset=DS, config_file_list=config_file_list)
    init_logger(config)
    logger = getLogger()

    # 记录当前设置
    logger.info(config)

    # plz set algo='exhaustive' to use exhaustive search, in this case, max_evals is auto set
    # 设置自动调参实例，传入待调参数和整体参数
    # 参数设置会进入到objective_function里
    hp = HyperTuning(objective_function, algo='exhaustive',
                     params_file=args.params_file, fixed_config_file_list=config_file_list)
    # 运行自动调参
    hp.run()

    # 输出最佳参数和最佳性能指标
    hp.export_result(output_file=args.output_file)
    logger.info(set_color('args ', 'yellow') + f': {args}')
    logger.info(set_color('best param ', 'yellow') + f': {hp.best_params}')
    logger.info(set_color('best result ', 'yellow') + f': {hp.params2result[hp.params2str(hp.best_params)]}')
    # print('best params: ', hp.best_params)
    # print('best result: ')
    # print(hp.params2result[hp.params2str(hp.best_params)])


if __name__ == '__main__':
    parameter_tuning()
