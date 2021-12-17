import os

def get_project_root():
    """
    Get the root path of SeqDes
    :rtype: str
    """
    return os.path.split(os.path.abspath(__file__))[0]+"/../.."

def get_model_path():
    """
    Get the path to the Jade Database
    :rtype: str
    """
    return get_project_root()+"/models"


