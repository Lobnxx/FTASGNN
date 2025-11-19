"""Main program for training and evaluation."""

import torch
import warnings
import random
import numpy as np
import dgl
import argparse
from .config import Config
from .models import create_model
from .data import DataLoader
from .training import Trainer
from .samplers import create_sampler
import logging
from .utils.logger import setup_logging
import os
from .utils.visualization import get_all_latent_representations, visualize_latent_space, visualize_raw_data
# from .adversarial.attacks import AdversarialAttacker # Replaced with AdversarialManager
# from .adversarial.utils import FlowData # FlowData is used indirectly via AdversarialManager
from .adversarial.manager import AdversarialManager # Import AdversarialManager
# import logging # Remove duplicate import
# from collections import defaultdict # defaultdict is no longer used directly in main

def parse_args():
    """Parses command line arguments."""
    parser = argparse.ArgumentParser(description='Traffic Detection Training')
    parser.add_argument("--seed", type=int, default=Config.SEED)
    parser.add_argument('--dataset', type=int, default=Config.DATASET_SELECT,
                      help='Dataset selection (0: MCFP, 1: USTC-TFC2016-master)')
    parser.add_argument('--model', type=int, default=Config.MODELS_SELECT,
                      help='Model selection (0: CombinedModel, 1: GraphSAGE, 2: CNN1D)')
    parser.add_argument('--sampler', type=str, default=Config.SAMPLER_TYPE,
                      help='Sampler type (bandit or neighbor)')
    parser.add_argument('--epoches', type=int, default=Config.EPOCHS,
                      help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=Config.BATCH_SIZE,
                      help='Training batch size')
    parser.add_argument('--lr', type=float, default=Config.LEARNING_RATE,
                      help='Learning rate')
    parser.add_argument('-b', type=bool, default=Config.BINARY_CLASS,
                        help="Set to True for binary classification, False for multi-class.")
    # --- Adversarial Attack Parameters ---
    parser.add_argument('--attack', type=bool, default=Config.USE_ADVERSARIAL_ATTACKS,
                        help='Whether to enable adversarial attacks.')
    parser.add_argument('--adv_attack_types', nargs='*', default=Config.ADVERSARIAL_ATTACK_TYPES,
                        help='List of adversarial attack types to apply (e.g., timing_jitter, padding).')
    parser.add_argument('--jitter_delta_ms', type=float, default=Config.JITTER_DELTA_MS,
                        help='Delta_ms for Timing Jitter Attack.')
    parser.add_argument('--padding_rho', type=float, default=Config.PADDING_RHO,
                        help='Rho for Packet Padding Attack.')
    parser.add_argument('--padding_p_max', type=int, default=Config.PADDING_P_MAX,
                        help='P_max for Packet Padding Attack.')
    parser.add_argument('--padding_mtu', type=int, default=Config.PADDING_MTU,
                        help='MTU for Packet Padding Attack.')
    parser.add_argument('--mimicry_alpha', type=float, default=Config.MIMICRY_ALPHA,
                        help='Alpha for Benign Mimicry Attack.')

    return parser.parse_args()

def set_seed(seed):
    """Sets random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    dgl.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    """Main function."""
    # Parse command line arguments
    args = parse_args()
    
    # Update configuration
    Config.DATASET_SELECT = args.dataset
    Config.MODELS_SELECT = args.model
    Config.SAMPLER_TYPE = args.sampler
    Config.EPOCHS = args.epoches
    Config.BATCH_SIZE = args.batch_size
    Config.LEARNING_RATE = args.lr
    Config.USE_KNN = args.use_knn
    Config.USE_SESSION_FLOW  = args.use_session
    Config.BINARY_CLASS = args.b

    # --- Update adversarial attack configuration ---
    Config.USE_ADVERSARIAL_ATTACKS = args.attack
    Config.ADVERSARIAL_ATTACK_TYPES = args.adv_attack_types
    Config.JITTER_DELTA_MS = args.jitter_delta_ms
    Config.PADDING_RHO = args.padding_rho
    Config.PADDING_P_MAX = args.padding_p_max
    Config.PADDING_MTU = args.padding_mtu
    Config.MIMICRY_ALPHA = args.mimicry_alpha

    # Set up logger
    setup_logging()
    # Create visualization directories
    vis_dir = os.path.join(Config.get_data_dir(), f"len_{Config.d_session}\\visualizations")
    raw_vis_dir = os.path.join(Config.get_data_dir(), "visualizations")
    os.makedirs(vis_dir, exist_ok=True)
    os.makedirs(raw_vis_dir, exist_ok=True)
    Config.VIS_DIR = vis_dir
    Config.RAW_VIS_DIR = raw_vis_dir
    # Log training configuration
    training_config = [
        ("Dataset", Config.DATASET_NAMES[Config.DATASET_SELECT]),
        ("Sampler", Config.SAMPLER_TYPE),
        ("Epochs", Config.EPOCHS),
        ("Batch Size", Config.BATCH_SIZE),
        ("Learning Rate", Config.LEARNING_RATE),
        ("Device", Config.DEVICE)
    ]
    # Add adversarial attack configuration to log
    if Config.USE_ADVERSARIAL_ATTACKS:
        training_config.append(("\nAdversarial Attacks Enabled", True))
        training_config.append(("\nAttack Types", Config.ADVERSARIAL_ATTACK_TYPES))
        if "timing_jitter" in Config.ADVERSARIAL_ATTACK_TYPES:
            training_config.append(("Jitter Delta (ms)", Config.JITTER_DELTA_MS))
        if "padding" in Config.ADVERSARIAL_ATTACK_TYPES:
            training_config.append(("Padding Rho", Config.PADDING_RHO))
            training_config.append(("Padding P_max", Config.PADDING_P_MAX))
            training_config.append(("Padding MTU", Config.PADDING_MTU))
        if "mimicry" in Config.ADVERSARIAL_ATTACK_TYPES:
            training_config.append(("Mimicry Alpha", Config.MIMICRY_ALPHA))

    training_config_str = "  ".join([f"{k}: {v}" for k, v in training_config])
    
    # Set random seed
    set_seed(args.seed)
    warnings.filterwarnings("ignore", category=UserWarning)
    try:
        logging.info("Initializing data loader...")
        data_loader = DataLoader()
        result = data_loader.load_data()
        all_vectors, all_labels, five_tuples, time_stamps = (
            result if data_loader.set_time_stamps else (*result, None)
        )
        
        # Convert raw data to a list of FlowData instances
        all_flow_data = None
        if Config.SESSION_PREFIXS[Config.prefix_num] == "P_SA":
            all_flow_data = DataLoader.convert_raw_data_to_flow_data(
            all_vectors, all_labels, five_tuples, time_stamps
        ) 
            # Separate malicious and benign flow data for Mimicry Attack's benign_pool
            malicious_flow_data = [fd for fd in all_flow_data if fd.label == 1]
            benign_flow_data = [fd for fd in all_flow_data if fd.label == 0]

            # Instantiate adversarial manager
            adversarial_manager = AdversarialManager(benign_flow_data=benign_flow_data)

        # --- Adversarial attack logic (currently commented out) ---
        if Config.USE_ADVERSARIAL_ATTACKS and all_flow_data is not None: # User control main switch
            logging.info("Applying flow-level adversarial attacks (currently commented out)...")
            all_flow_data = adversarial_manager.apply_attacks_to_flows(all_flow_data)
            
            # Convert attacked FlowData back to original data format
            if all_flow_data:
                all_vectors = np.array([fd.vectors for fd in all_flow_data])
                all_labels = np.array([fd.label for fd in all_flow_data])
                five_tuples = [fd.five_tuple for fd in all_flow_data]
                time_stamps = np.array([fd.timestamps for fd in all_flow_data])
                logging.info("Converted attacked FlowData back to raw data format.")
            else:
                logging.warning("No flows were attacked or processed. Using original data.")

        # Build graph
        g = data_loader.build_graph(all_vectors, five_tuples)
        # Add node features and labels
        g = data_loader._add_node_features(g, all_vectors, all_labels, time_stamps)
        
        model = create_model(Config.MODELS_SELECT)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY
        )
        training_config_str += f"  model: {model.name}"
        start_flag = '\n'+"-"*(9)+"Training Config"+"-"*(9) +'\n'
        end_flag = '\n'+"-"*(len(start_flag)-2)
        logging.info(start_flag+training_config_str+end_flag)

        # Create sampler
        sampler = create_sampler(
            Config.SAMPLER_TYPE,
            fanouts=[10, 10],
            train_mask=g.ndata['train_mask'],
            device=Config.DEVICE
        )
        
        # Create trainer
        trainer = Trainer(model, g, sampler, optimizer)
        
        # --- Model loading logic ---
        if not Config.train:
            dataset_name = Config.get_dataset_name()
            if Config.USE_ADVERSARIAL_ATTACKS:
                dataset_name += f"_adv_{Config.MN}"
            model_name = Config.MODEL_NAMES[Config.MODELS_SELECT]
            load_model_path = os.path.join(Config.MODEL_SAVE_DIR, f"{dataset_name}/{model_name}/best.pt")
            logging.info(f"Loading pre-trained model from: {load_model_path}")
            try:
                trainer.load_model_checkpoint(load_model_path)
                logging.info("Model loaded successfully.")
            except FileNotFoundError as e:
                logging.error(f"Error loading model: {e}. Exiting.")
                return 
            except Exception as e:
                logging.error(f"An unexpected error occurred while loading model: {e}", exc_info=True)
                return 

        # Start training
        visualize_raw_data(
            data=g.ndata['feat'][g.ndata['test_mask']].clone().detach().cpu().numpy(),
            labels=g.ndata['label'][g.ndata['test_mask']].clone().detach().cpu().numpy(),
            n_components=Config.OUT_CHANNELS,
            save_path=Config.RAW_VIS_DIR
        )
        
        trainer.train_epochs(train=Config.train)
        
        # Get latent representations of all data and visualize
        logging.info("Generating latent representations...")
        latent_reps, labels = get_all_latent_representations(model, sampler, g)
        
        # Visualize with PCA
        logging.info("Visualizing with PCA...")
        visualize_latent_space(
            latent_reps, 
            labels,
            method='pca',
            n_components=Config.OUT_CHANNELS,
            save_path=Config.VIS_DIR
        )
    except Exception as e:
        logging.error(f"Error during training: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()
