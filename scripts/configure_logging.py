import logging
import torch
import shutil


def configure_logging(log_filename):
    logger = logging.getLogger(log_filename)
    # logger = logging.getLogger("my_logger")
    logger.setLevel(logging.DEBUG)
    # Format for our log lines
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    # Setup console logging
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # Setup file logging
    fh = logging.FileHandler(log_filename)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger


def save_checkpoint(state, is_best, epoch, checkpoint='checkpoint', filename='checkpoint.pth.tar'):
    filePath = checkpoint + filename + 'epoch:' + str(epoch)
    torch.save(state, filePath)
    if is_best:
        best_filePath = checkpoint + filename + 'best'
        shutil.copyfile(filePath, best_filePath)


def get_learning_rate(optimizer):
    lr = []
    for param_group in optimizer.param_groups:
        lr += [param_group['lr']]
    return lr
