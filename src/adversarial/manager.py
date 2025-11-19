import logging
import numpy as np
import dgl
import torch
from typing import List, Optional, Tuple, Any
from ..config import Config
from .attacks import AdversarialAttacker, graph_poison_attack
from .defenses import timestamp_denoise, size_denoise, augment_with_adversarial_samples, ensemble_predict
from .utils import FlowData
import copy
import random
class AdversarialManager:
    """
    Manages adversarial attacks and defenses.
    Encapsulates the logic for attacks and defenses, providing a unified API for the main function.
    """
    def __init__(self, benign_flow_data: List[FlowData]):
        self.logger = logging.getLogger(__name__)
        self.attacker = AdversarialAttacker(benign_pool=benign_flow_data)
        # If defenses are needed, defense-related classes or functions can be initialized here
        # self.defender = AdversarialDefender(...) 
        self.benign_flow_data = benign_flow_data

    def apply_attacks_to_flows(self, all_flow_data: List[FlowData]) -> List[FlowData]:
        """
        Applies selected attacks to the FlowData list based on configuration.
        :param all_flow_data: Original list of FlowData instances.
        :return: List of FlowData instances after attacks.
        """
        if not Config.USE_ADVERSARIAL_ATTACKS or not Config.ADVERSARIAL_ATTACK_TYPES:
            self.logger.info("Adversarial attacks are disabled or no attack types specified.")
            return all_flow_data

        self.logger.info(f"Applying adversarial attacks: {Config.ADVERSARIAL_ATTACK_TYPES}")
        attacked_flow_data_list = []

        k = max(1, int(len(all_flow_data) * Config.MR))  # 2%
        indices = random.sample(range(len(all_flow_data)), k)

        for i, flow_data in enumerate(all_flow_data):
            current_flow_data = copy.deepcopy(flow_data) # Attack a copy of each flow
            if i not in indices:
                attacked_flow_data_list.append(current_flow_data)
                continue
            if "timing_jitter" in Config.ADVERSARIAL_ATTACK_TYPES:
                current_flow_data = self.attacker.timing_jitter(current_flow_data, Config.JITTER_DELTA_MS)
                # self.logger.debug(f"Applied Timing Jitter Attack to flow with label {flow_data.label}")

            if "padding" in Config.ADVERSARIAL_ATTACK_TYPES:
                current_flow_data = self.attacker.padding(current_flow_data, Config.PADDING_RHO, Config.PADDING_P_MAX, Config.PADDING_MTU)
                # self.logger.debug(f"Applied Packet Padding Attack to flow with label {flow_data.label}")

            if "mimicry" in Config.ADVERSARIAL_ATTACK_TYPES and current_flow_data.label == 1: # Mimicry is only effective for malicious flows
                current_flow_data = self.attacker.mimicry(current_flow_data, Config.MIMICRY_ALPHA)
                # self.logger.debug(f"Applied Benign Mimicry Attack to malicious flow")
            
            attacked_flow_data_list.append(current_flow_data)
        
        self.logger.info("Flow-level adversarial attacks applied.")
        return attacked_flow_data_list
