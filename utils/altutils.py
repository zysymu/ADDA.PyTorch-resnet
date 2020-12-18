import os
import torch
from torchvision import datasets, transforms
import torch.utils.data as data
import configparser
import logging


def readConfigFile(filePath):
    """
    Read config file

    Args:
        filePath ([str]): path to config file

    Returns:
        [Obj]: config object
    """    
    config = configparser.ConfigParser()
    config.read(filePath)
    return config


def setLogger(logFilePath):
    """
    Set logger

    Args:
        logFilePath ([str]): path to log file

    Returns:
        [obj]: logger object
    """    
    logHandler = [logging.FileHandler(logFilePath), logging.StreamHandler()]
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", handlers=logHandler)
    logger = logging.getLogger()
    return logger

def get_office(dataset_root, batch_size, category):
    """Get Office datasets loader

    Args:
        dataset_root (str): path to the dataset folder
        batch_size (int): batch size
        category (str): category of Office dataset (amazon, webcam, dslr)

    Returns:
        obj: dataloader object for Office dataset
    """    
    # image pre-processing
    pre_process = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])

    # datasets and data_loader
    office_dataset = datasets.ImageFolder(
        os.path.join(dataset_root, 'office31', category, 'images'), transform=pre_process)

    office_dataloader = torch.utils.data.DataLoader(
        dataset=office_dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)

    return office_dataloader