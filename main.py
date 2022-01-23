import logging
import torch

from exp import ExpModelTrainScratch, ExpModelTrainSISA, ExpMemInfScratch, ExpMemInfSISA
from parameter_parser import parameter_parser


def config_logger():
    # create logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # create console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    # create formatter and add it to the handlers
    formatter = logging.Formatter('%(levelname)s:%(asctime)s: - %(name)s - : %(message)s')
    ch.setFormatter(formatter)

    # add the handlers to the logger
    logger.addHandler(ch)


def main(args):
    torch.cuda.set_device(args["cuda"])
    if args['exp'] == "model_train" and args["unlearning_method"] == "scratch":
        ExpModelTrainScratch(args)
    elif args['exp'] == "model_train" and args["unlearning_method"] == "sisa":
        ExpModelTrainSISA(args)
    elif args['exp'] == "mem_inf" and args["unlearning_method"] == "scratch":
        ExpMemInfScratch(args)
    elif args['exp'] == "mem_inf" and args["unlearning_method"] == "sisa":
        ExpMemInfSISA(args)
    else:
        raise Exception("invalid exp name or unlearning method name")


if __name__ == "__main__":
    args = parameter_parser()
    config_logger()
    main(args)
