import argparse

import hydra


def train(cfg):
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser("MLOpsSampleTrain")
    parser.add_argument_group("--projectname",
                              "-p",
                              help="directory name in which hydra config file is searched.")
    args = parser.parse_args()

    project_name = args.projectname
