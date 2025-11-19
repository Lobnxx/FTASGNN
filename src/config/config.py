import torch
import os
from typing import Union
class Config:

    # 0: MCFP, 1: USTC-TFC2016, 2: CIRA-CIC-DoHBrw-2020
    DATASET_NAMES = ["MCFP", "USTC-TFC2016-master"]
    DATASET_SELECT = 0

    MODEL_NAMES = ["GraphSAGE", "CNN1D", "TA-PGCN"]
    MODELS_SELECT = 2
    
    OUT_CHANNELS = 2
    #### very importtant
    BINARY_CLASS = True
    HIDDEN_DIM = 128

    prefix_num = 1
    SESSION_PREFIXS = ["T_SA", "P_SA"]
    data_dir_name = f"{SESSION_PREFIXS[prefix_num]}_traffic_data"
    id_to_label:Union[dict, None] = None
    label_to_id:Union[dict, None] = None
    VIS_DIR:Union[str, None] = None
    RAW_VIS_DIR:Union[str, None] = None

    heads = 2
    seq_len = 7
    d_packet = 64
    d_session = seq_len*d_packet
    

    train = False
    MR = 1
    MN = 0
    EPOCHS = 10
    USE_SESSION_FLOW = 1 
    USE_KNN = 1  
    SAMPLER_TYPE = "mask"
    # TEST_TIMES_PER_POINT = 10 if SAMPLER_TYPE == "bandit" else 1
    BATCH_SIZE = 1024
    LEARNING_RATE = 0.0001
    WEIGHT_DECAY = 5e-4
    SEED = 3407
    K = 1
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    CHECKPOINT_DIR = os.path.join(os.getcwd(), 'checkpoints')
    PATIENCE = EPOCHS

    MODEL_SAVE_DIR = os.path.join(os.getcwd(), 'saved_models')

    USE_ADVERSARIAL_ATTACKS = True 
    ALL_ATTACK_TYPES = ["timing_jitter", "padding", "mimicry"]
    ADVERSARIAL_ATTACK_TYPES = ALL_ATTACK_TYPES[:3]
    

    JITTER_DELTA_MS = 200 # Timing Jitter Attack  delta_ms
    PADDING_RHO = 0.3 # Packet Padding Attack   rho
    PADDING_P_MAX = 300 # Packet Padding Attack   p_max
    PADDING_MTU = 1500 # Packet Padding Attack   MTU
    MIMICRY_ALPHA = 0.5 # Benign Mimicry Attack   alpha
    GRAPH_POISON_K = 20 # Graph Poisoning Attack   k
    
    @classmethod
    def get_dataset_name(cls):
        return cls.DATASET_NAMES[cls.DATASET_SELECT]
    
    @classmethod
    def get_raw_data_dir(cls):
        data_dir = cls.get_data_dir()
        print(f"Using dataset: {data_dir}")
        raw_data_dir = os.path.join(data_dir, "raw_data")
        if not os.path.isdir(raw_data_dir):
            raise FileNotFoundError(f"Raw data directory {raw_data_dir} does not exist.")
        return raw_data_dir

    @classmethod
    def get_data_dir(cls):
        data_dir = os.path.join(f"dataset\\{cls.get_dataset_name()}", cls.data_dir_name)
        if not os.path.isdir(data_dir):
            raise FileNotFoundError(f"Data directory {data_dir} does not exist.")
        return data_dir
