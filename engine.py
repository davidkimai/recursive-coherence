# recursive_entropy_manager/core/engine.py

import numpy as np
import torch
import torch.nn as nn
import math
import logging
from typing import Dict, List, Tuple, Optional, Union, Any

from .symbolic_residue import SymbolicResidueTensor
from .phase_alignment import PhaseAlignmentDetector
from .coherence import CoherenceMeasurementEngine
from .attractor import AttractorStabilizer
from .metabolism import ContradictionMetabolismEngine
from .beverly_band import BeverlyBandCalculator
from .constants import *

logger = logging.getLogger(__name__)


class RecursiveEntropyManager:
    """
    Core implementation of the Recursive Entropy Manager (REM).
    
    This class integrates all components of the REM system into a unified 
    framework that can be applied to any transformer architecture.
    
    Parameters:
        model_config (Dict): Configuration of the transformer model
        rem_config (Dict): Configuration for the REM system
    """
    
    def __init__(self, model_config: Dict[str, Any], rem_config: Dict[str, Any]):
        self.model_config = model_config
        self.rem_config = rem_config
        
        # Initialize core components
        self.symbolic_residue_tracker = SymbolicResidueTensor(
            num_layers=model_config["num_layers"],
            num_heads=model_config["num_heads"],
            hidden_dim=model_config["hidden_dim"],
            config=rem_config.get("symbolic_residue", {})
        )
        
        self.phase_detector = PhaseAlignmentDetector(
            hidden_dim=model_config["hidden_dim"],
            config=rem_config.get("phase_alignment", {})
        )
        
        self.coherence_engine = CoherenceMeasurementEngine(
            model_config=model_config,
            config=rem_config.get("coherence", {})
        )
        
        self.attractor_system = AttractorStabilizer(
            hidden_dim=model_config["hidden_dim"],
            config=rem_config.get("attractor", {})
        )
        
        self.metabolism_engine = ContradictionMetabolismEngine(
            hidden_dim=model_config["hidden_dim"],
            config=rem_config.get("metabolism", {})
        )
        
        self.beverly_band = BeverlyBandCalculator(
            config=rem_config.get("beverly_band", {})
        )
        
        # Internal state
        self.layer_states = {}
        self.global_state = {
            "total_steps": 0,
            "coherence_history": [],
            "residue_history": [],
            "phase_history": [],
            "attractor_history": [],
            "metabolism_history": [],
            "beverly_band_history": []
        }
        
        logger.info("Recursive Entropy Manager initialized")
    
    def calculate_compression_coefficient(self, num_tokens: int, bandwidth: float) -> float:
        """
        Calculate the Recursive Compression Coefficient (γ).
        
        Parameters:
            num_tokens (int): Number of tokens or recursive operations
            bandwidth (float): Available information bandwidth
            
        Returns:
            float: Compression coefficient γ
        """
        return math.log(num_tokens / bandwidth + 1)
    
    def calculate_attractor_strength(self, num_tokens: int, gamma: float) -> float:
        """
        Calculate the Attractor Activation Strength A(N).
        
        Parameters:
            num_tokens (int): Number of tokens or recursive operations
            gamma (float): Compression coefficient
            
        Returns:
            float: Attractor strength A(N)
        """
        return 1.0 - (gamma / num_tokens)
    
    def process_layer(self, 
                     layer_idx: int, 
                     input_states: torch.Tensor, 
                     output_states: torch.Tensor,
