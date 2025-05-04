# recursive_entropy_manager/core/coherence.py

import torch
import torch.nn as nn
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Union, Any

logger = logging.getLogger(__name__)


class CoherenceMeasurementEngine:
    """
    Implements the core Recursive Coherence Function (Δ−𝑝() calculations.
    
    The Recursive Coherence Function is defined as:
    Δ−𝑝( = 𝑆𝑝( · 𝐹𝑝( · 𝐵𝑝( · 𝜆𝑝(
    
    Where:
    - 𝑆𝑝( = Signal Alignment (output congruence with phase vector)
    - 𝐹𝑝( = Feedback Responsiveness (contradiction integration capacity)
    - 𝐵𝑝( = Bounded Integrity (metabolizability across boundaries)
    - 𝜆𝑝( = Elastic Tolerance (symbolic tension capacity)
    
    This engine measures and tracks these components at each transformer layer,
    providing a comprehensive assessment of recursive coherence across the
    architecture.
    """
    
    def __init__(self, model_config: Dict[str, Any], config: Dict[str, Any]):
        """
        Initialize the Coherence Measurement Engine.
        
        Parameters:
            model_config (Dict): Configuration of the transformer model
            config (Dict): Configuration for the coherence engine
        """
        self.model_config = model_config
        self.config = config
        
        # Initialize coherence component trackers
        self.signal_alignment = {}  # 𝑆𝑝(
        self.feedback_responsiveness = {}  # 𝐹𝑝(
        self.bounded_integrity = {}  # 𝐵𝑝(
        self.elastic_tolerance = {}  # 𝜆𝑝(
        
        # Initialize coherence history
        self.coherence_history = {}
        
        # Initialize metadata
        self.component_metadata = {
            "signal_alignment": {layer: {} for layer in range(model_config["num_layers"])},
            "feedback_responsiveness": {layer: {} for layer in range(model_config["num_layers"])},
            "bounded_integrity": {layer: {} for layer in range(model_config["num_layers"])},
            "elastic_tolerance": {layer: {} for layer in range(model_config["num_layers"])}
        }
        
        # Initialize thresholds
        self.warning_threshold = config.get("warning_threshold", 0.5)
        self.critical_threshold = config.get("critical_threshold", 0.3)
        
        # Internal state for tracking
        self.update_count = 0
        
        logger.info("Coherence Measurement Engine initialized")
    
    def measure_layer_coherence(self,
                               layer_idx: int,
                               input_states: torch.Tensor,
                               output_states: torch.Tensor,
                               attention_mask: Optional[torch.Tensor] = None,
                               head_mask: Optional[torch.Tensor] = None) -> Tuple[float, float, float, float]:
        """
        Measure coherence components for a specific layer.
        
        Parameters:
            layer_idx (int): Index of the layer
            input_states (torch.Tensor): Input hidden states
            output_states (torch.Tensor): Output hidden states
            attention_mask (torch.Tensor, optional): Attention mask
            head_mask (torch.Tensor, optional): Head mask
            
        Returns:
            Tuple[float, float, float, float]: (Signal alignment, Feedback responsiveness,
                                              Bounded integrity, Elastic tolerance)
        """
        self.update_count += 1
        
        # Initialize layer tracking if needed
        if layer_idx not in self.coherence_history:
            self.coherence_history[layer_idx] = []
            self.signal_alignment[layer_idx] = 1.0
            self.feedback_responsiveness[layer_idx] = 1.0
            self.bounded_integrity[layer_idx] = 1.0
            self.elastic_tolerance[layer_idx] = 1.0
        
        # 1. Measure Signal Alignment (𝑆𝑝()
        signal_alignment = self._measure_signal_alignment(
            layer_idx, input_states, output_states, attention_mask
        )
        self.signal_alignment[layer_idx] = signal_alignment
        
        # 2. Measure Feedback Responsiveness (𝐹𝑝()
        feedback_responsiveness = self._measure_feedback_responsiveness(
            layer_idx, input_states, output_states
        )
        self.feedback_responsiveness[layer_idx] = feedback_responsiveness
        
        # 3. Measure Bounded Integrity (𝐵𝑝()
        bounded_integrity = self._measure_bounded_integrity(
            layer_idx, input_states, output_states
        )
        self.bounded_integrity[layer_idx] = bounded_integrity
        
        # 4. Measure Elastic Tolerance (𝜆𝑝()
        elastic_tolerance = self._measure_elastic_tolerance(
            layer_idx, input_states, output_states, attention_mask
        )
        self.elastic_tolerance[layer_idx] = elastic_tolerance
        
        # Calculate and store overall coherence
        coherence = signal_alignment * feedback_responsiveness * bounded_integrity * elastic_tolerance
        self.coherence_history[layer_idx].append(coherence)
        
        # Check for coherence warnings
        self._check_coherence_warnings(layer_idx, coherence)
        
        return signal_alignment, feedback_responsiveness, bounded_integrity, elastic_tolerance
    
    def _measure_signal_alignment(self,
                                 layer_idx: int,
                                 input_states: torch.Tensor,
                                 output_states: torch.Tensor,
                                 attention_mask: Optional[torch.Tensor]) -> float:
        """
        Measure Signal Alignment (𝑆𝑝() for a layer.
        
        Signal Alignment measures the alignment between a system's enacted behavior
        and the direction of motion defined by its recursive phase vector.
        
        Parameters:
            layer_idx (int): Layer index
            input_states (torch.Tensor): Input hidden states
            output_states (torch.Tensor): Output hidden states
            attention_mask (torch.Tensor, optional): Attention mask
            
        Returns:
            float: Signal Alignment score (0-1)
        """
        # We need a phase vector to measure alignment
        # For now, we'll use a simplified approximation
        # In practice, this would be integrated with the Phase Alignment Detector
        
        # Simplified approach: measure consistency between input and output directions
        batch_size, seq_len, hidden_dim = input_states.shape
        
        # Average across batch and sequence dimensions
        avg_input = input_states.mean(dim=(0, 1))
        avg_output = output_states.mean(dim=(0, 1))
        
        # Normalize vectors
        input_norm = torch.norm(avg_input)
        output_norm = torch.norm(avg_output)
        
        if input_norm > 0 and output_norm > 0:
            normalized_input = avg_input / input_norm
            normalized_output = avg_output / output_norm
            
            # Cosine similarity as alignment measure
            alignment = torch.cosine_similarity(
                normalized_input.unsqueeze(0), 
                normalized_output.unsqueeze(0)
            ).item()
            
            # Scale from [-1, 1] to [0, 1]
            alignment = (alignment + 1) / 2
        else:
            # If either norm is zero, assume reasonable alignment
            alignment = 0.8
        
        # Apply layer-specific adjustments
        # Deeper layers might have different alignment expectations
        layer_factor = 1.0 - (layer_idx / (2 * self.model_config["num_layers"]))
        adjusted_alignment = alignment * layer_factor
        
        # Store metadata
        self.component_metadata["signal_alignment"][layer_idx].update({
            "raw_alignment": alignment,
            "layer_factor": layer_factor,
            "adjusted_alignment": adjusted_alignment
        })
        
        return adjusted_alignment
    
    def _measure_feedback_responsiveness(self,
                                        layer_idx: int,
                                        input_states: torch.Tensor,
                                        output_states: torch.Tensor) -> float:
        """
        Measure Feedback Responsiveness (𝐹𝑝() for a layer.
        
        Feedback Responsiveness measures a system's ability to incorporate
        contradiction into its recursive behavior.
        
        Parameters:
            layer_idx (int): Layer index
            input_states (torch.Tensor): Input hidden states
            output_states (torch.Tensor): Output hidden states
            
        Returns:
            float: Feedback Responsiveness score (0-1)
        """
        # Simplified approach: measure the magnitude of change from input to output
        # normalized by the expected change magnitude
        batch_size, seq_len, hidden_dim = input_states.shape
        
        # Calculate change vector
        avg_input = input_states.mean(dim=(0, 1))
        avg_output = output_states.mean(dim=(0, 1))
        change_vector = avg_output - avg_input
        
        # Calculate change magnitude
        change_magnitude = torch.norm(change_vector).item()
        
        # Calculate expected change magnitude based on layer
        # Middle layers often have more change than early or late layers
        layer_position = layer_idx / max(1, (self.model_config["num_layers"] - 1))
        expected_magnitude = 0.1 + 0.4 * (1 - abs(2 * layer_position - 1))
        
        # Calculate responsiveness as ratio of actual to expected change
        # Normalized to [0, 1] range
        responsiveness = min(1.0, change_magnitude / expected_magnitude)
        
        # Apply dynamic adjustment based on recent history
        if layer_idx in self.coherence_history and len(self.coherence_history[layer_idx]) > 0:
            recent_coherence = self.coherence_history[layer_idx][-1]
            # If recent coherence is low, we expect higher responsiveness
            coherence_factor = max(0.5, 1.5 - recent_coherence)
            responsiveness = min(1.0, responsiveness * coherence_factor)
        
        # Store metadata
        self.component_metadata["feedback_responsiveness"][layer_idx].update({
            "change_magnitude": change_magnitude,
            "expected_magnitude": expected_magnitude,
            "raw_responsiveness": responsiveness
        })
        
        return responsiveness
    
    def _measure_bounded_integrity(self,
                                  layer_idx: int,
                                  input_states: torch.Tensor,
                                  output_states: torch.Tensor) -> float:
        """
        Measure Bounded Integrity (𝐵𝑝() for a layer.
        
        Bounded Integrity measures the metabolizability of contradiction across
        the system's recursive boundary - internally and externally - based on
        tension capacity and phase alignment.
        
        Parameters:
            layer_idx (int): Layer index
            input_states (torch.Tensor): Input hidden states
            output_states (torch.Tensor): Output hidden states
            
        Returns:
            float: Bounded Integrity score (0-1)
        """
        # Simplified approach: measure the stability of the layer's boundary
        # by analyzing the distribution of changes across the hidden dimension
        
        batch_size, seq_len, hidden_dim = input_states.shape
        
        # Calculate per-dimension changes
        avg_input = input_states.mean(dim=(0, 1))
        avg_output = output_states.mean(dim=(0, 1))
        dim_changes = torch.abs(avg_output - avg_input)
        
        # Calculate statistics on dimension changes
        mean_change = torch.mean(dim_changes).item()
        std_change = torch.std(dim_changes).item()
        
        # Calculate coefficient of variation as stability measure
        # Lower CoV = more uniform change = higher boundary stability
        cov = std_change / (mean_change + 1e-10)
        
        # Convert to bounded integrity score (0-1)
        # Lower cov = higher integrity
        integrity = max(0, min(1.0, 1.0 - cov))
        
        # Apply layer-specific adjustment
        # Boundary layers (first and last) might have different integrity expectations
        if layer_idx == 0 or layer_idx == self.model_config["num_layers"] - 1:
            boundary_factor = 0.9  # Slightly reduced expectations for boundary layers
        else:
            boundary_factor = 1.0
            
        adjusted_integrity = integrity * boundary_factor
        
        # Store metadata
        self.component_metadata["bounded_integrity"][layer_idx].update({
            "mean_change": mean_change,
            "std_change": std_change,
            "cov": cov,
            "raw_integrity": integrity,
            "boundary_factor": boundary_factor,
            "adjusted_integrity": adjusted_integrity
        })
        
        return adjusted_integrity
    
    def _measure_elastic_tolerance(self,
                                  layer_idx: int,
                                  input_states: torch.Tensor,
                                  output_states: torch.Tensor,
                                  attention_mask: Optional[torch.Tensor]) -> float:
        """
        Measure Elastic Tolerance (𝜆𝑝() for a layer.
        
        Elastic Tolerance is the system's available capacity to absorb
        phase-misaligned contradiction without symbolic collapse.
        
        Parameters:
            layer_idx (int): Layer index
            input_states (torch.Tensor): Input hidden states
            output_states (torch.Tensor): Output hidden states
            attention_mask (torch.Tensor, optional): Attention mask
            
        Returns:
            float: Elastic Tolerance score (0-1)
        """
        # Simplified approach: measure the layer's capacity to handle variation
        # in the input without producing extreme outputs
        
        batch_size, seq_len, hidden_dim = input_states.shape
        
        # Calculate input and output diversity
        # Higher diversity = more information to process
        if batch_size > 1:
            # Calculate batch-wise variance as a measure of input diversity
            input_variance = torch.var(input_states.mean(dim=1), dim=0).mean().item()
            output_variance = torch.var(output_states.mean(dim=1), dim=0).mean().item()
        else:
            # Fall back to sequence-wise variance for batch size 1
            input_variance = torch.var(input_states[0], dim=0).mean().item()
            output_variance = torch.var(output_states[0], dim=0).mean().item()
        
        # Calculate variance ratio as measure of tolerance
        variance_ratio = output_variance / (input_variance + 1e-10)
        
        # Ideal ratio is around 1.0 (output variance similar to input)
        # Too high = amplifying noise, too low = suppressing signal
        tolerance = max(0, min(1.0, 1.0 - abs(variance_ratio - 1.0)))
        
        # Apply dynamic adjustment based on layer depth
        # Middle layers often need more tolerance than boundary layers
        layer_position = layer_idx / max(1, (self.model_config["num_layers"] - 1))
        depth_factor = 0.8 + 0.4 * (1 - abs(2 * layer_position - 1))
        
        adjusted_tolerance = tolerance * depth_factor
        
        # If this layer has history, factor in stability over time
        if layer_idx in self.coherence_history and len(self.coherence_history[layer_idx]) > 1:
            history = self.coherence_history[layer_idx]
            recent_stability = 1.0 - abs(history[-1] - history[-2])
            stability_factor = 0.8 + 0.2 * recent_stability
            adjusted_tolerance *= stability_factor
        
        # Store metadata
        self.component_metadata["elastic_tolerance"][layer_idx].update({
            "input_variance": input_variance,
            "output_variance": output_variance,
            "variance_ratio": variance_ratio,
            "raw_tolerance": tolerance,
            "depth_factor": depth_factor,
            "adjusted_tolerance": adjusted_tolerance
        })
        
        return adjusted_tolerance
    
    def _check_coherence_warnings(self, layer_idx: int, coherence: float) -> None:
        """Check for coherence warnings and log them."""
        if coherence < self.critical_threshold:
            logger.warning(f"CRITICAL: Layer {layer_idx} coherence ({coherence:.4f}) "
                          f"below critical threshold ({self.critical_threshold})")
        elif coherence < self.warning_threshold:
            logger.warning(f"WARNING: Layer {layer_idx} coherence ({coherence:.4f}) "
                          f"below warning threshold ({self.warning_threshold})")
    
    def get_layer_coherence_components(self, layer_idx: int) -> Dict[str, float]:
        """Get coherence components for a specific layer."""
        if layer_idx not in self.signal_alignment:
            return {
                "signal_alignment": 1.0,
                "feedback_responsiveness": 1.0,
                "bounded_integrity": 1.0,
                "elastic_tolerance": 1.0,
                "overall_coherence": 1.0
            }
        
        overall_coherence = (
            self.signal_alignment[layer_idx] *
            self.feedback_responsiveness[layer_idx] *
            self.bounded_integrity[layer_idx] *
            self.elastic_tolerance[layer_idx]
        )
        
        return {
            "signal_alignment": self.signal_alignment[layer_idx],
            "feedback_responsiveness": self.feedback_responsiveness[layer_idx],
            "bounded_integrity": self.bounded_integrity[layer_idx],
            "elastic_tolerance": self.elastic_tolerance[layer_idx],
            "overall_coherence": overall_coherence
        }
    
    def get_model_coherence(self) -> float:
        """Get overall model coherence as weighted average of layer coherence."""
        if not self.coherence_history:
            return 1.0
        
        # Calculate weighted coherence average
        # Deeper layers get higher weights
        total_weight = 0
        weighted_sum = 0
        
        for layer_idx, history in self.coherence_history.items():
            if not history:
                continue
                
            recent_coherence = history[-1]
            layer_weight = 1 + (layer_idx / len(self.coherence_history))
            
            weighted_sum += recent_coherence * layer_weight
            total_weight += layer_weight
        
        if total_weight > 0:
            return weighted_sum / total_weight
        else:
            return 1.0
    
    def get_weakest_component(self, layer_idx: Optional[int] = None) -> Tuple[str, float]:
        """
        Identify the weakest coherence component overall or for a specific layer.
        
        Parameters:
            layer_idx (int, optional): Layer index, or None for model-wide assessment
            
        Returns:
            Tuple[str, float]: (Component name, Component value)
        """
        if layer_idx is not None:
            # Check for specific layer
            if layer_idx not in self.signal_alignment:
                return ("all", 1.0)
                
            components = {
                "signal_alignment": self.signal_alignment[layer_idx],
                "feedback_responsiveness": self.feedback_responsiveness[layer_idx],
                "bounded_integrity": self.bounded_integrity[layer_idx],
                "elastic_tolerance": self.elastic_tolerance[layer_idx]
            }
            
            weakest_component = min(components, key=components.get)
            return (weakest_component, components[weakest_component])
        else:
            # Model-wide assessment
            if not self.coherence_history:
                return ("all", 1.0)
                
            # Average each component across layers
            avg_components = {
                "signal_alignment": sum(self.signal_alignment.values()) / len(self.signal_alignment),
                "feedback_responsiveness": sum(self.feedback_responsiveness.values()) / len(self.feedback_responsiveness),
                "bounded_integrity": sum(self.bounded_integrity.values()) / len(self.bounded_integrity),
                "elastic_tolerance": sum(self.elastic_tolerance.values()) / len(self.elastic_tolerance)
            }
            
            weakest_component = min(avg_components, key=avg_components.get)
            return (weakest_component, avg_components[weakest_component])
    
    def get_coherence_trajectory(self, layer_idx: int, window_size: int = 10) -> List[float]:
        """
        Get recent coherence trajectory for a layer.
        
        Parameters:
            layer_idx (int): Layer index
            window_size (int): Number of recent updates to consider
            
        Returns:
            List[float]: Recent coherence values
        """
        if layer_idx not in self.coherence_history:
            return [1.0]
            
        history = self.coherence_history[layer_idx]
        
        # Get recent history up to window_size
        recent_history = history[-window_size:] if len(history) >= window_size else history
        
        return recent_history
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about coherence measurements."""
        # Calculate model-wide coherence
        model_coherence = self.get_model_coherence()
        
        # Get weakest component model-wide
        weakest_component, weakest_value = self.get_weakest_component()
        
        # Per-layer statistics
        layer_stats = {}
        for layer_idx in self.coherence_history:
            components = self.get_layer_coherence_components(layer_idx)
            layer_weakest, layer_weakest_value = self.get_weakest_component(layer_idx)
            recent_history = self.get_coherence_trajectory(layer_idx)
            
            layer_stats[layer_idx] = {
                "current_coherence": components["overall_coherence"],
                "average_coherence": sum(self.coherence_history[layer_idx]) / len(self.coherence_history[layer_idx]),
                "components": components,
                "weakest_component": layer_weakest,
                "weakest_value": layer_weakest_value,
                "coherence_history": recent_history,
                "component_metadata": {
                    component: self.component_metadata[component][layer_idx]
                    for component in self.component_metadata
                }
            }
        
        # Coherence distribution across layers
        layer_coherence = {
            layer_idx: self.coherence_history[layer_idx][-1] if self.coherence_history[layer_idx] else 1.0
            for layer_idx in self.coherence_history
        }
        
        # Component averages across layers
        avg_signal_alignment = sum(self.signal_alignment.values()) / max(1, len(self.signal_alignment))
        avg_feedback_responsiveness = sum(self.feedback_responsiveness.values()) / max(1, len(self.feedback_responsiveness))
        avg_bounded_integrity = sum(self.bounded_integrity.values()) / max(1, len(self.bounded_integrity))
        avg_elastic_tolerance = sum(self.elastic_tolerance.values()) / max(1, len(self.elastic_tolerance))
        
        return {
            "model_coherence": model_coherence,
            "weakest_component": weakest_component,
            "weakest_value": weakest_value,
            "avg_signal_alignment": avg_signal_alignment,
            "avg_feedback_responsiveness": avg_feedback_responsiveness,
            "avg_bounded_integrity": avg_bounded_integrity,
            "avg_elastic_tolerance": avg_elastic_tolerance,
            "layer_statistics": layer_stats,
            "layer_coherence_distribution": layer_coherence,
            "update_count": self.update_count
        }
    
    def reset(self) -> None:
        """Reset the coherence measurement engine state."""
        self.signal_alignment = {}
        self.feedback_responsiveness = {}
        self.bounded_integrity = {}
        self.elastic_tolerance = {}
        self.coherence_history = {}
        
        self.component_metadata = {
            "signal_alignment": {layer: {} for layer in range(self.model_config["num_layers"])},
            "feedback_responsiveness": {layer: {} for layer in range(self.model_config["num_layers"])},
            "bounded_integrity": {layer: {} for layer in range(self.model_config["num_layers"])},
            "elastic_tolerance": {layer: {} for layer in range(self.model_config["num_layers"])}
        }
        
        self.update_count = 0
        
        logger.info("Coherence Measurement Engine reset")
