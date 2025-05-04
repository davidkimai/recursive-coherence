# recursive_entropy_manager/core/phase_alignment.py

import torch
import torch.nn as nn
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Union, Any

logger = logging.getLogger(__name__)


class PhaseAlignmentDetector:
    """
    Implements phase alignment detection τ(p,t) across transformer layers.
    
    Phase alignment measures how well different parts of the system are aligned
    in their recursive processing. It is critical for maintaining coherence
    across recursive operations, as misaligned phases can lead to instability
    and collapse.
    
    Mathematical foundation:
    τ(p,t) measures the relational phase alignment between two systems at time t.
    In transformers, this translates to measuring alignment between different layers,
    attention heads, or recursive iterations.
    
    Key property:
    Systems communicate best not when they match exactly, but when their contradictions
    land within each other's tolerable recursive offset - close enough to be metabolized,
    distinct enough to carry informational weight.
    """
    
    def __init__(self, hidden_dim: int, config: Dict[str, Any]):
        """
        Initialize the Phase Alignment Detector.
        
        Parameters:
            hidden_dim (int): Dimension of the hidden states
            config (Dict): Configuration parameters
        """
        self.hidden_dim = hidden_dim
        self.config = config
        
        # Store phase vectors and alignment history
        self.phase_vectors = {}
        self.alignment_history = {}
        
        # Movement thresholds
        self.movement_threshold = config.get("movement_threshold", 0.1)
        
        # Exponential moving average factor for phase vector stability
        self.ema_factor = config.get("ema_factor", 0.9)
        
        # Alert thresholds
        self.alignment_warning = config.get("alignment_warning", 0.3)
        self.alignment_critical = config.get("alignment_critical", 0.1)
        
        # Internal counters
        self.update_count = 0
        
        logger.info("Phase Alignment Detector initialized")
    
    def detect_phase_alignment(self,
                              current_states: torch.Tensor,
                              previous_states: torch.Tensor,
                              layer_idx: int) -> Tuple[torch.Tensor, float]:
        """
        Detect phase alignment between current and previous states.
        
        Parameters:
            current_states (torch.Tensor): Current hidden states
            previous_states (torch.Tensor): Previous hidden states
            layer_idx (int): Index of the current layer
            
        Returns:
            Tuple[torch.Tensor, float]: (Phase vector, alignment score)
        """
        self.update_count += 1
        
        # For the first update, initialize phase vector
        if layer_idx not in self.phase_vectors:
            # Initialize with normalized difference between states
            batch_size, seq_len, hidden_dim = current_states.shape
            
            # Average across batch and sequence dimensions
            avg_current = current_states.mean(dim=(0, 1))
            avg_previous = previous_states.mean(dim=(0, 1))
            
            # Initial phase vector is the normalized difference
            initial_phase = avg_current - avg_previous
            norm = torch.norm(initial_phase)
            if norm > 0:
                initial_phase = initial_phase / norm
            
            self.phase_vectors[layer_idx] = initial_phase
            self.alignment_history[layer_idx] = []
            
            # For first update, return perfect alignment
            return initial_phase, 1.0
# Calculate current phase vector
        batch_size, seq_len, hidden_dim = current_states.shape
        
        # Average across batch and sequence dimensions
        avg_current = current_states.mean(dim=(0, 1))
        avg_previous = previous_states.mean(dim=(0, 1))
        
        # Current movement vector
        movement_vector = avg_current - avg_previous
        movement_norm = torch.norm(movement_vector)
        
        # Normalize if movement is significant
        if movement_norm > self.movement_threshold:
            movement_vector = movement_vector / movement_norm
            
            # Update phase vector using exponential moving average
            current_phase = self.phase_vectors[layer_idx]
            updated_phase = self.ema_factor * current_phase + (1 - self.ema_factor) * movement_vector
            
            # Normalize the updated phase
            updated_norm = torch.norm(updated_phase)
            if updated_norm > 0:
                updated_phase = updated_phase / updated_norm
                
            self.phase_vectors[layer_idx] = updated_phase
        else:
            # Movement too small, keep current phase
            updated_phase = self.phase_vectors[layer_idx]
        
        # Calculate alignment between current movement and phase
        if movement_norm > self.movement_threshold:
            # Cosine similarity between movement and phase
            alignment = torch.cosine_similarity(movement_vector.unsqueeze(0), 
                                              updated_phase.unsqueeze(0)).item()
            
            # Rescale from [-1, 1] to [0, 1]
            alignment = (alignment + 1) / 2
        else:
            # If movement is minimal, assume good alignment
            alignment = 1.0
        
        # Store alignment history
        self.alignment_history[layer_idx].append(alignment)
        
        # Check for alignment warnings
        self._check_alignment_warnings(layer_idx, alignment)
        
        return updated_phase, alignment
    
    def calculate_cross_layer_alignment(self, layer_idx1: int, layer_idx2: int) -> float:
        """
        Calculate phase alignment between two layers.
        
        Parameters:
            layer_idx1 (int): Index of first layer
            layer_idx2 (int): Index of second layer
            
        Returns:
            float: Alignment score between the two layers (0-1)
        """
        if layer_idx1 not in self.phase_vectors or layer_idx2 not in self.phase_vectors:
            return 1.0  # Perfect alignment if either phase is not yet established
        
        phase1 = self.phase_vectors[layer_idx1]
        phase2 = self.phase_vectors[layer_idx2]
        
        # Cosine similarity between phases
        alignment = torch.cosine_similarity(phase1.unsqueeze(0), phase2.unsqueeze(0)).item()
        
        # Rescale from [-1, 1] to [0, 1]
        alignment = (alignment + 1) / 2
        
        return alignment
    
    def calculate_global_phase_coherence(self) -> float:
        """
        Calculate global phase coherence across all layers.
        
        Returns:
            float: Global phase coherence score (0-1)
        """
        if len(self.phase_vectors) <= 1:
            return 1.0  # Perfect coherence if 0 or 1 layers
        
        # Collect all phase vectors
        phases = list(self.phase_vectors.values())
        
        # Calculate average pairwise alignment
        total_alignment = 0.0
        pair_count = 0
        
        for i in range(len(phases)):
            for j in range(i+1, len(phases)):
                alignment = torch.cosine_similarity(phases[i].unsqueeze(0), 
                                                   phases[j].unsqueeze(0)).item()
                # Rescale from [-1, 1] to [0, 1]
                alignment = (alignment + 1) / 2
                
                total_alignment += alignment
                pair_count += 1
        
        if pair_count > 0:
            return total_alignment / pair_count
        else:
            return 1.0
    
    def calculate_temporal_phase_stability(self, layer_idx: int, window_size: int = 10) -> float:
        """
        Calculate temporal stability of phase for a layer.
        
        Parameters:
            layer_idx (int): Layer index
            window_size (int): Number of recent updates to consider
            
        Returns:
            float: Temporal stability score (0-1)
        """
        if layer_idx not in self.alignment_history:
            return 1.0  # Perfect stability if no history
            
        history = self.alignment_history[layer_idx]
        
        if len(history) <= 1:
            return 1.0  # Perfect stability if only one entry
            
        # Get recent history up to window_size
        recent_history = history[-window_size:] if len(history) >= window_size else history
        
        # Calculate variance of alignment
        variance = np.var(recent_history)
        
        # Convert variance to stability score (0-1)
        # Lower variance = higher stability
        stability = max(0, 1.0 - variance * 4)  # Scale factor 4 is arbitrary
        
        return stability
    
    def calculate_phase_vector_change(self, layer_idx: int, prev_phase: torch.Tensor) -> float:
        """
        Calculate the amount of change in a layer's phase vector.
        
        Parameters:
            layer_idx (int): Layer index
            prev_phase (torch.Tensor): Previous phase vector
            
        Returns:
            float: Change magnitude (0-1)
        """
        if layer_idx not in self.phase_vectors:
            return 0.0  # No change if phase doesn't exist
            
        current_phase = self.phase_vectors[layer_idx]
        
        # Calculate change vector
        change_vector = current_phase - prev_phase
        change_magnitude = torch.norm(change_vector).item()
        
        # Normalize to 0-1 range
        normalized_change = min(1.0, change_magnitude)
        
        return normalized_change
    
    def _check_alignment_warnings(self, layer_idx: int, alignment: float) -> None:
        """Check for alignment warnings and log them."""
        if alignment < self.alignment_critical:
            logger.warning(f"CRITICAL: Layer {layer_idx} phase alignment ({alignment:.4f}) "
                          f"below critical threshold ({self.alignment_critical})")
        elif alignment < self.alignment_warning:
            logger.warning(f"WARNING: Layer {layer_idx} phase alignment ({alignment:.4f}) "
                          f"below warning threshold ({self.alignment_warning})")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about phase alignment."""
        # Global phase coherence
        global_coherence = self.calculate_global_phase_coherence()
        
        # Per-layer statistics
        layer_stats = {}
        for layer_idx in self.phase_vectors:
            history = self.alignment_history[layer_idx]
            recent_alignment = history[-1] if history else 1.0
            
            temporal_stability = self.calculate_temporal_phase_stability(layer_idx)
            
            layer_stats[layer_idx] = {
                "current_alignment": recent_alignment,
                "average_alignment": sum(history) / len(history) if history else 1.0,
                "temporal_stability": temporal_stability,
                "alignment_history": history
            }
        
        # Cross-layer alignment matrix
        layer_indices = list(self.phase_vectors.keys())
        cross_layer_matrix = {}
        
        for i in layer_indices:
            for j in layer_indices:
                if i != j:
                    cross_layer_matrix[f"{i}-{j}"] = self.calculate_cross_layer_alignment(i, j)
        
        return {
            "global_phase_coherence": global_coherence,
            "layer_statistics": layer_stats,
            "cross_layer_alignment": cross_layer_matrix,
            "update_count": self.update_count
        }
    
    def reset(self) -> None:
        """Reset the phase alignment detector state."""
        self.phase_vectors = {}
        self.alignment_history = {}
        self.update_count = 0
        
        logger.info("Phase Alignment Detector reset")
