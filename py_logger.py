def init_logger(log_file):
    from logging import getLogger, INFO, FileHandler, Formatter, StreamHandler
    logger = getLogger(__name__)
    logger.setLevel(INFO)
    handler1 = StreamHandler()
    handler1.setFormatter(Formatter("%(message)s"))
    handler2 = FileHandler(filename=log_file)
    handler2.setFormatter(Formatter("%(message)s"))
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    return logger
  
# def set_seed(seed=None, cudnn_deterministic=True):
#   if seed is None:
#       seed = 42

#   os.environ['PYTHONHASHSEED'] = str(seed)
#   np.random.seed(seed)
#   random.seed(seed)
#   torch.manual_seed(seed)
#   torch.cuda.manual_seed(seed)
#   torch.backends.cudnn.deterministic = cudnn_deterministic
#   torch.backends.cudnn.benchmark = False
    
Logger = init_logger(log_file=CFG.log_path)    
