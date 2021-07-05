import argparse
from train import main as train
from test import main as test
import os
import glob
import shutil

# Paths to files that are generated from training/testing to move into the results folder for each run
RESULTS_FILES = ['results.txt', 'results.png', 'precision.txt', 'recall.txt', 'PR_curve.png', 'test_results.txt']


class Conifg:
    def __init__(self):
        pass

    def save(self, path):
        print(self.__dict__)
        with open(path, 'w') as f:
            for key, value in self.__dict__.items():
                f.write(f'{key}: {value}\n')


class TrainingConifg(Conifg):
    def __init__(self,
                 epochs=300,
                 batch_size=8,
                 cfg='cfg/yolov3-spp.cfg',
                 data=None,
                 multi_scale=False,
                 img_size=[608, 608],
                 rect=False,
                 resume=False,
                 nosave=False,
                 notest=False,
                 evolve=False,
                 bucket='',
                 cache_images=False,
                 weights='weights/yolov3-spp-ultralytics.pt',
                 name='',
                 device='0',
                 adam=True,
                 single_cls=False,
                 freeze_layers=False,
                 results_file='results.txt'):
        self.epochs = epochs
        self.batch_size = batch_size
        self.cfg = cfg
        self.data = data
        self.multi_scale = multi_scale
        self.img_size = img_size
        self.rect = rect
        self.resume = resume
        self.nosave = nosave
        self.notest = notest
        self.evolve = evolve
        self.bucket = bucket
        self.cache_images = cache_images
        self.weights = weights
        self.name = name
        self.device = device
        self.adam = adam
        self.single_cls = single_cls
        self.freeze_layers = freeze_layers
        self.results_file = results_file


class TestConfig(Conifg):
    def __init__(self,
                 cfg='cfg/yolov3-spp.cfg',
                 data=None,
                 weights='weights/last.pt',
                 batch_size=8,
                 img_size=608,
                 conf_thres=0.0001,
                 iou_thres=0.6,
                 save_json=False,
                 task='test',
                 device='0',
                 single_cls=False,
                 augment=False):
        self.cfg = cfg
        self.data = data
        self.weights = weights
        self.batch_size = batch_size
        self.img_size = img_size
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.save_json = save_json
        self.task = task
        self.device = device
        self.single_cls = single_cls
        self.augment = augment


def main(args):
    runs = args.runs
    experiments_folder = args.experiments_folder
    results_dir = args.results_dir
    assert os.path.isdir(experiments_folder), f'Experiments folder {experiments_folder} is not a valid directory'

    if not os.path.exists(results_dir):
        os.mkdir(results_dir)

    experiments = os.listdir(experiments_folder)
    for experiment in experiments:
        experiment_results_folder = os.path.join(results_dir, experiment)
        if not os.path.exists(experiment_results_folder):
            os.mkdir(experiment_results_folder)

        conditions = os.listdir(os.path.join(experiments_folder, experiment))
        for condition in conditions:
            data = glob.glob(os.path.join(experiments_folder, experiment, condition, '*.data'))[0]

            condition_results_folder = os.path.join(experiment_results_folder, condition)
            if not os.path.exists(condition_results_folder):
                os.mkdir(condition_results_folder)

            for run in range(runs):
                run_results_folder = os.path.join(condition_results_folder, f'run-{run}')
                if not os.path.exists(run_results_folder):
                    os.mkdir(run_results_folder)

                print(f'---------------- Running training for {run_results_folder} ----------------')
                train_config = TrainingConifg()
                train_config.data = data
                train_config.save(os.path.join(run_results_folder, 'train_config.txt'))
                train(opt=train_config)

                print(f'---------------- Running testing for {run_results_folder} ----------------')
                test_config = TestConfig()
                test_config.data = data
                train_config.save(os.path.join(run_results_folder, 'test_config.txt'))
                test(opt=test_config)

                # After training and testing, move the results to a unique folder so they don't get overwritten
                for src in RESULTS_FILES:
                    if not os.path.isfile(src):
                        print(f'Warning: could not find {src} to copy into results file')
                        continue
                    dst = os.path.join(run_results_folder, os.path.basename(src))
                    shutil.move(src, dst)

            # TODO generate summary PR curve for all runs for each condition


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiments-folder', type=str, default='experiment_setup',
                        help='Experiments folder generated by setup. Will run each')
    parser.add_argument('--runs', type=int, default=4, help='Number of runs to complete for each condition')
    parser.add_argument('--results-dir', type=str, default='experiment_results',
                        help='Path to directory to save training and testing results')
    args = parser.parse_args()
    main(args)
