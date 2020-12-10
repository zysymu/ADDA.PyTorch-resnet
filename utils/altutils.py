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