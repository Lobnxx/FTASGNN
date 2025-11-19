import numpy as np
import random
import networkx as nx
from sklearn.neighbors import NearestNeighbors
from .utils import FlowData # Import FlowData
from ..config import Config # Import Config
from typing import List, Optional, Tuple

def _adjust_vectors_for_padding(original_vectors: np.ndarray,
                                raw_frame_lengths: np.ndarray,
                                new_lengths: np.ndarray,
                                max_packet_length: int,
                                ) -> np.ndarray:
    """
    Adjusts vectors based on raw_frame_length + padding_length.
    Only pads random bytes of padding_length after raw_frame_length.
    """
    adjusted = np.copy(original_vectors)

    raw = raw_frame_lengths.astype(int)
    new = new_lengths.astype(int)

    # Limit not to exceed max_packet_length
    raw = np.clip(raw, 0, max_packet_length)
    new = np.clip(new, 0, max_packet_length)

    # Step 1: First clear the part after new_len
    for i in range(len(new)):
        adjusted[i, new[i]:max_packet_length] = 0

    # Step 2: Pad the raw_len → new_len part
    for i in range(len(raw)):
        if new[i] > raw[i]:
            size = new[i] - raw[i]
            adjusted[i, raw[i]:new[i]] = np.random.randint(
                1, 256, size=size, dtype=np.uint8
            )

    return adjusted


def timing_jitter_attack(flow_data: FlowData, delta_ms: float) -> np.ndarray:
    """
    Adds Uniform noise to packet timestamps in FlowData.
    :param flow_data: Original FlowData instance.
    :param delta_ms: Perturbation range in milliseconds.
    :return: Perturbed timestamp sequence.
    """
    if flow_data.timestamps is None or len(flow_data.timestamps) == 0:
        return flow_data.timestamps
    noise = np.random.uniform(-delta_ms / 1000, delta_ms / 1000, size=len(flow_data.timestamps))
    return flow_data.timestamps + noise

def padding_attack(flow_data: FlowData, rho: float, p_max: int, mtu: int,
                   include_session_padding_frames: bool = True):
    """
    Applies padding attack to frames of a session.
    
    Returns:
        padding_lengths: np.ndarray, length consistent with original frames,
                         actual amount added per frame (0 where not attacked).
        raw_frame_lengths: np.ndarray, true effective length of frames before attack (trailing zeros removed).
    """

    if flow_data.packet_lengths is None or len(flow_data.packet_lengths) == 0:
        return None, None

    packet_lengths = flow_data.packet_lengths
    frames = flow_data.vectors  # shape: (num_frames, 64)
    n_total_frames = len(packet_lengths)

    # ================================
    # 1) Calculate original raw_frame_lengths
    # ================================
    raw_frame_lengths = np.zeros(n_total_frames, dtype=int)

    for i in range(n_total_frames):
        frame = frames[i]
        trailing_zeros = 0
        # Count consecutive zeros from the end
        for x in reversed(frame):
            if x == 0:
                trailing_zeros += 1
            else:
                break
        raw_frame_lengths[i] = 64 - trailing_zeros

    # ================================
    # 2) Select attackable frames
    # ================================
    if include_session_padding_frames:
        attackable_indices = np.arange(n_total_frames)
    else:
        attackable_indices = np.arange(flow_data.actual_packet_count)

    if len(attackable_indices) == 0:
        return np.zeros(n_total_frames), raw_frame_lengths

    num_frames_to_modify = int(len(attackable_indices) * rho)
    if num_frames_to_modify == 0:
        return np.zeros(n_total_frames), raw_frame_lengths

    idx_to_modify = np.random.choice(attackable_indices, num_frames_to_modify, replace=False)

    # ================================
    # 3) Generate padding amount
    # ================================
    padding_lengths = np.zeros(n_total_frames)

    pad = np.random.randint(0, p_max, size=len(idx_to_modify))

    # Final padding added = min(original_len + pad, mtu) - original_len
    for k, frame_idx in enumerate(idx_to_modify):
        # original_len = packet_lengths[frame_idx]
        padding_lengths[frame_idx] = min(Config.d_packet - raw_frame_lengths[frame_idx], pad[k])

    return padding_lengths, raw_frame_lengths

def mimicry_attack(mal_flow_data: FlowData, ben_flow_data: FlowData, alpha: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Implements mixed time intervals & packet sizes & vectors according to the formula F_adv = α * F_mal + (1 - α) * F_ben.
    :param mal_flow_data: FlowData instance of malicious flow.
    :param ben_flow_data: FlowData instance of benign flow.
    :param alpha: Mixing ratio.
    :return: Tuple of mixed (timestamps, packet_lengths, vectors).
    """
    # Ensure both have timestamps and packet lengths
    if mal_flow_data.timestamps is None or ben_flow_data.timestamps is None or \
       mal_flow_data.packet_lengths is None or ben_flow_data.packet_lengths is None or \
       mal_flow_data.vectors is None or ben_flow_data.vectors is None:
        print("Warning: Missing timestamps, packet_lengths or vectors for mimicry attack. Returning original malicious flow features.")
        return mal_flow_data.timestamps, mal_flow_data.packet_lengths, mal_flow_data.vectors

    mal_timestamps = np.array(mal_flow_data.timestamps)
    ben_timestamps = np.array(ben_flow_data.timestamps)
    mal_packet_lengths = np.array(mal_flow_data.packet_lengths)
    ben_packet_lengths = np.array(ben_flow_data.packet_lengths)
    mal_vectors = np.array(mal_flow_data.vectors)
    ben_vectors = np.array(ben_flow_data.vectors)

    # Ensure consistent length, use the shortest
    L = min(len(mal_timestamps), len(ben_timestamps))
    mal_timestamps = mal_timestamps[:L]
    ben_timestamps = ben_timestamps[:L]
    mal_packet_lengths = mal_packet_lengths[:L]
    ben_packet_lengths = ben_packet_lengths[:L]
    mal_vectors = mal_vectors[:L, :]
    ben_vectors = ben_vectors[:L, :]

    mixed_timestamps = alpha * mal_timestamps + (1 - alpha) * ben_timestamps
    mixed_packet_lengths = alpha * mal_packet_lengths + (1 - alpha) * ben_packet_lengths
    mixed_vectors = alpha * mal_vectors + (1 - alpha) * ben_vectors # Mixing byte data, may require further processing

    return mixed_timestamps, mixed_packet_lengths, mixed_vectors.astype(np.uint8)


class AdversarialAttacker:
    """
    Unified attack API.
    """
    def __init__(self, benign_pool: Optional[List[FlowData]] = None):
        """
        :param benign_pool: Pool of benign flows for Mimicry Attack.
                            Each flow should be a FlowData instance.
        """
        self.benign_pool = benign_pool if benign_pool is not None else []

    def timing_jitter(self, flow_data: FlowData, delta_ms: float) -> FlowData:
        """
        Applies Timing Jitter Attack.
        :param flow_data: Original FlowData instance.
        :param delta_ms: Perturbation range in milliseconds.
        :return: Perturbed FlowData instance.
        """
        perturbed_flow_data = flow_data.copy()
        perturbed_flow_data.timestamps = timing_jitter_attack(flow_data, delta_ms)
        return perturbed_flow_data

    def padding(self, flow_data: FlowData, rho: float, p_max: int, mtu: int) -> FlowData:
        """
        Applies Packet Padding Attack.
        :param flow_data: Original FlowData instance.
        :param rho: Perturbation ratio.
        :param p_max: Maximum padding size.
        :param mtu: Maximum Transmission Unit.
        :return: Perturbed FlowData instance.
        """
        if flow_data.packet_lengths is None:
            print("Warning: packet_lengths not available for padding attack. Skipping.")
            return flow_data
            
        perturbed_flow_data = flow_data.copy()
        original_packet_lengths = flow_data.packet_lengths.copy() # Save original lengths for adjusting vectors
        
        padding_lengths, raw_frame_lengths = padding_attack(flow_data, rho, p_max, mtu)
        perturbed_flow_data.packet_lengths = padding_lengths + raw_frame_lengths
        
        # Adjust vectors based on new packet_lengths
        if flow_data.vectors is not None:
            max_packet_length = flow_data.vectors.shape[1] if flow_data.vectors.ndim > 1 else 0
            perturbed_flow_data.vectors = _adjust_vectors_for_padding(flow_data.vectors,
                                                                      raw_frame_lengths,
                                                                      perturbed_flow_data.packet_lengths,
                                                                      max_packet_length)
        return perturbed_flow_data

    def mimicry(self, mal_flow_data: FlowData, alpha: float) -> FlowData:
        """
        Applies Benign Mimicry Attack.
        :param mal_flow_data: FlowData instance of malicious flow.
        :param alpha: Mixing ratio.
        :return: Mixed FlowData instance.
        """
        if not self.benign_pool:
            print("Warning: Benign pool is empty for mimicry attack. Returning original malicious flow.")
            return mal_flow_data

        ben_flow_data = random.choice(self.benign_pool)
        
        mixed_timestamps, mixed_packet_lengths, mixed_vectors = mimicry_attack(mal_flow_data, ben_flow_data, alpha)
        
        perturbed_flow_data = mal_flow_data.copy()
        perturbed_flow_data.timestamps = mixed_timestamps
        perturbed_flow_data.packet_lengths = mixed_packet_lengths
        perturbed_flow_data.vectors = mixed_vectors
        
        return perturbed_flow_data
