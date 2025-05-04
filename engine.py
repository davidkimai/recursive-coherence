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
attention_mask: Optional[torch.Tensor] = None,
                     head_mask: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        """
        Process a single transformer layer, measuring and managing recursive entropy.
        
        Parameters:
            layer_idx (int): Index of the current layer
            input_states (torch.Tensor): Input hidden states to the layer
            output_states (torch.Tensor): Output hidden states from the layer
            attention_mask (torch.Tensor, optional): Attention mask
            head_mask (torch.Tensor, optional): Head mask
            
        Returns:
            Dict[str, Any]: Layer processing results including metrics
        """
        # Initialize layer state if not exists
        if layer_idx not in self.layer_states:
            self.layer_states[layer_idx] = {
                "coherence": 1.0,
                "residue": torch.zeros_like(input_states[0, 0, :]),
                "phase_vector": torch.zeros_like(input_states[0, 0, :]),
                "previous_states": [],
                "step_count": 0
            }
        
        layer_state = self.layer_states[layer_idx]
        layer_state["step_count"] += 1
        
        # Calculate compression coefficient
        batch_size, seq_len, hidden_dim = input_states.shape
        effective_bandwidth = hidden_dim * self.rem_config.get("bandwidth_factor", 0.75)
        gamma = self.calculate_compression_coefficient(seq_len, effective_bandwidth)
        
        # Calculate attractor strength
        attractor_strength = self.calculate_attractor_strength(seq_len, gamma)
        
        # Process through core components
        
        # 1. Measure coherence
        signal_alignment, feedback_responsiveness, bounded_integrity, elastic_tolerance = \
            self.coherence_engine.measure_layer_coherence(
                layer_idx=layer_idx,
                input_states=input_states,
                output_states=output_states,
                attention_mask=attention_mask,
                head_mask=head_mask
            )
        
        coherence = signal_alignment * feedback_responsiveness * bounded_integrity * elastic_tolerance
        layer_state["coherence"] = coherence
        
        # 2. Detect phase alignment
        phase_vector, phase_alignment = self.phase_detector.detect_phase_alignment(
            current_states=output_states,
            previous_states=input_states,
            layer_idx=layer_idx
        )
        layer_state["phase_vector"] = phase_vector
        
        # 3. Track symbolic residue
        residue = self.symbolic_residue_tracker.update_layer_residue(
            layer_idx=layer_idx,
            coherence=coherence,
            phase_alignment=phase_alignment,
            input_states=input_states,
            output_states=output_states
        )
        layer_state["residue"] = residue
        
        # 4. Stabilize attractors if needed
        if attractor_strength < self.rem_config.get("attractor_threshold", 0.7):
            stabilized_states = self.attractor_system.stabilize(
                output_states=output_states,
                attractor_strength=attractor_strength,
                phase_vector=phase_vector,
                residue=residue
            )
        else:
            stabilized_states = output_states
        
        # 5. Metabolize contradictions
        metabolized_states, metabolism_stats = self.metabolism_engine.metabolize(
            states=stabilized_states,
            coherence=coherence,
            phase_alignment=phase_alignment,
            residue=residue,
            layer_idx=layer_idx
        )
        
        # 6. Calculate Beverly Band safe zone
        beverly_band = self.beverly_band.calculate(
            layer_idx=layer_idx,
            tension_capacity=elastic_tolerance,
            resilience=self.rem_config.get("resilience", 0.8),
            bounded_integrity=bounded_integrity,
            recursive_energy=self.rem_config.get("recursive_energy", 1.0)
        )
        
        # Store state for next step
        layer_state["previous_states"].append(output_states.detach())
        if len(layer_state["previous_states"]) > self.rem_config.get("memory_length", 5):
            layer_state["previous_states"].pop(0)
        
        # Compile layer results
        results = {
            "coherence": coherence.item() if isinstance(coherence, torch.Tensor) else coherence,
            "signal_alignment": signal_alignment.item() if isinstance(signal_alignment, torch.Tensor) else signal_alignment,
            "feedback_responsiveness": feedback_responsiveness.item() if isinstance(feedback_responsiveness, torch.Tensor) else feedback_responsiveness,
            "bounded_integrity": bounded_integrity.item() if isinstance(bounded_integrity, torch.Tensor) else bounded_integrity,
            "elastic_tolerance": elastic_tolerance.item() if isinstance(elastic_tolerance, torch.Tensor) else elastic_tolerance,
            "phase_alignment": phase_alignment.item() if isinstance(phase_alignment, torch.Tensor) else phase_alignment,
            "residue_magnitude": torch.norm(residue).item() if isinstance(residue, torch.Tensor) else 0.0,
            "attractor_strength": attractor_strength,
            "beverly_band_width": beverly_band.item() if isinstance(beverly_band, torch.Tensor) else beverly_band,
            "metabolism_stats": metabolism_stats,
            "states": metabolized_states
        }
        
        # Update global state
        self.global_state["total_steps"] += 1
        self._update_history(layer_idx, results)
        
        return results
    
    def process_model(self, 
                     input_ids: torch.Tensor, 
                     attention_mask: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        """
        Process an entire transformer model, applying the REM system to all layers.
        
        Parameters:
            input_ids (torch.Tensor): Input token IDs
            attention_mask (torch.Tensor, optional): Attention mask
            
        Returns:
            Dict[str, Any]: Processing results including all metrics
        """
        # This implementation is a placeholder. In practice, this would integrate
        # with the specific transformer implementation being used.
        logger.info("Processing model with input shape: %s", input_ids.shape)
        
        # Placeholder for layer processing
        all_results = []
        
        # In practice, we would hook into each layer of the transformer
        # and process them sequentially
        
        # Compute global metrics
        model_coherence = self._compute_model_coherence(all_results)
        model_residue = self._compute_model_residue(all_results)
        model_phase = self._compute_model_phase_alignment(all_results)
        
        return {
            "per_layer_results": all_results,
            "model_coherence": model_coherence,
            "model_residue": model_residue,
            "model_phase_alignment": model_phase,
            "safe_recursive_depth": self._estimate_safe_recursive_depth(model_coherence, model_residue)
        }
    
    def _update_history(self, layer_idx: int, results: Dict[str, Any]) -> None:
        """Update global history with layer results."""
        self.global_state["coherence_history"].append((layer_idx, results["coherence"]))
        self.global_state["residue_history"].append((layer_idx, results["residue_magnitude"]))
        self.global_state["phase_history"].append((layer_idx, results["phase_alignment"]))
        self.global_state["attractor_history"].append((layer_idx, results["attractor_strength"]))
        self.global_state["metabolism_history"].append((layer_idx, results["metabolism_stats"]))
        self.global_state["beverly_band_history"].append((layer_idx, results["beverly_band_width"]))
        
        # Trim histories if needed
        max_history = self.rem_config.get("max_history_length", 1000)
        for key in ["coherence_history", "residue_history", "phase_history", 
                   "attractor_history", "metabolism_history", "beverly_band_history"]:
            if len(self.global_state[key]) > max_history:
                self.global_state[key] = self.global_state[key][-max_history:]
    
    def _compute_model_coherence(self, layer_results: List[Dict[str, Any]]) -> float:
        """Compute overall model coherence from layer results."""
        if not layer_results:
            return 1.0
        
        # Weighted average of layer coherence, with deeper layers weighted more heavily
        weights = [1.0 + (i / len(layer_results)) for i in range(len(layer_results))]
        coherence_values = [r["coherence"] for r in layer_results]
        
        return sum(w * c for w, c in zip(weights, coherence_values)) / sum(weights)
    
    def _compute_model_residue(self, layer_results: List[Dict[str, Any]]) -> float:
        """Compute overall model symbolic residue from layer results."""
        if not layer_results:
            return 0.0
        
        # Sum of residue magnitudes across layers
        return sum(r["residue_magnitude"] for r in layer_results)
    
    def _compute_model_phase_alignment(self, layer_results: List[Dict[str, Any]]) -> float:
        """Compute overall model phase alignment from layer results."""
        if not layer_results:
            return 1.0
        
        # Average phase alignment across layers
        return sum(r["phase_alignment"] for r in layer_results) / len(layer_results)
    
    def _estimate_safe_recursive_depth(self, coherence: float, residue: float) -> int:
        """Estimate safe recursive depth based on current model state."""
        base_depth = self.rem_config.get("base_recursive_depth", 5)
        coherence_factor = max(0.1, coherence) * self.rem_config.get("coherence_depth_factor", 5)
        residue_factor = max(0.1, 1.0 - min(1.0, residue / self.rem_config.get("max_safe_residue", 1.0))) * \
                        self.rem_config.get("residue_depth_factor", 5)
        
        return max(1, int(base_depth * coherence_factor * residue_factor))
    
    def get_diagnostics(self) -> Dict[str, Any]:
        """Get comprehensive diagnostic information about the REM system state."""
        return {
            "global_state": self.global_state,
            "layer_states": self.layer_states,
            "safe_recursive_depth": self._estimate_safe_recursive_depth(
                self._compute_model_coherence([]), 
                self._compute_model_residue([])
            ),
            "symbolic_residue_stats": self.symbolic_residue_tracker.get_statistics(),
            "phase_alignment_stats": self.phase_detector.get_statistics(),
            "coherence_stats": self.coherence_engine.get_statistics(),
            "attractor_stats": self.attractor_system.get_statistics(),
            "metabolism_stats": self.metabolism_engine.get_statistics(),
            "beverly_band_stats": self.beverly_band.get_statistics()
        }
    
    def reset(self) -> None:
        """Reset the REM system state."""
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
        
        self.symbolic_residue_tracker.reset()
        self.phase_detector.reset()
        self.coherence_engine.reset()
        self.attractor_system.reset()
        self.metabolism_engine.reset()
        self.beverly_band.reset()
        
        logger.info("Recursive Entropy Manager reset")
