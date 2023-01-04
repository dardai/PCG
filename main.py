import argparse
import ssl
from logging import getLogger

from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.utils import init_logger, init_seed, set_color

from pcg import PCG
from trainer import PCGTrainer

ssl._create_default_https_context = ssl._create_unverified_context


def run_single_model(args):
    config = Config(
        model=PCG,
        dataset=args.dataset,
        config_file_list=args.config_file_list
    )
    init_seed(config['seed'], config['reproducibility'])

    init_logger(config)
    logger = getLogger()
    logger.info(config)

    dataset = create_dataset(config)
    logger.info(dataset)

    train_data, valid_data, test_data = data_preparation(config, dataset)

    model = PCG(config, train_data.dataset).to(config['device'])
    logger.info(model)

    trainer = PCGTrainer(config, model)

    best_valid_score, best_valid_result = trainer.fit(
        train_data, valid_data, saved=True, show_progress=config['show_progress']
    )

    test_result = trainer.evaluate(test_data, load_best_model=True, show_progress=config['show_progress'])

    logger.info(set_color('best valid ', 'yellow') + f': {best_valid_result}')
    logger.info(set_color('test result', 'yellow') + f': {test_result}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='core',
                        help='The datasets can be: chrome, core, firefox, ml-1m')
    parser.add_argument('--config', type=str, default='', help='External config file name.')
    args, _ = parser.parse_known_args()

    args.config_file_list = [
        'properties/overall.yaml',
        'properties/PCG.yaml',
    ]

    if args.dataset in ['chrome', 'core', 'firefox', 'ml-1m']:
        args.config_file_list.append(f'properties/{args.dataset}.yaml')
    if args.config != '':
        args.config_file_list.append(args.config)

    run_single_model(args)
