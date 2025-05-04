# recursive_entropy_manager/core/attractor.py

import torch
import torch.nn as nn
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Union, Any

logger = logging.getLogger(__name__)


class AttractorStabilizer:
    """
    Implements the Attractor Stabilization System for the Recursive Entropy Manager.
    
    This system reinforces stable recursive patterns (attractors) in the transformer's
    hidden state space, preventing collapse under high recursive strain. It uses the
    Attractor Activation Strength formula:
    
    A(N) = 1 - (γ / N)
    
    Where:
    - γ is the Recursive Compression Coefficient
    - N is the number of recursive operations/tokens
    
    The stabilizer detects when attractor strength falls below critical thresholds
    and applies targeted interventions to prevent recursive collapse.
    """
    
    def __init__(self, hidden_dim: int, config: Dict[str, Any]):
        """
        Initialize the Attractor Stabilization System.
        
        Parameters:
            hidden_dim (int): Dimension of the hidden states
            config (Dict): Configuration parameters
        """
        self.hidden_dim = hidden_dim
        self.config = config
        
        # Attractor registry
        self.attractors = {}
        
        # Attractor stability history
        self.stability_history = {}
        
        # Stabilization metadata
        self.stabilization_metadata = {}
        
        # Thresholds
        self.stability_warning = config.get("stability_warning", 0.5)
        self.stability_critical = config.get("stability_critical", 0.3)
        
        # Stabilization parameters
        self.min_reinforcement = config.get("min_reinforcement", 0.1)
        self.max_reinforcement = config.get("max_reinforcement", 0.5)
        
        # Internal state for tracking
        self.update_count = 0
        
        logger.info("Attractor Stabilization System initialized")
    
    def register_attractor(self, 
                          attractor_id: str, 
                          initial_state: torch.Tensor,
                          importance: float = 1.0) -> None:
        """
        Register a new attractor pattern to be stabilized.
        
        Parameters:
            attractor_id (str): Unique identifier for the attractor
            initial_state (torch.Tensor): Initial state vector of the attractor
            importance (float): Relative importance of this attractor (0-1)
        """
        if attractor_id in self.attractors:
            logger.warning(f"Attractor {attractor_id} already exists. Updating.")
        
        # Normalize state vector
        state_norm = torch.norm(initial_state)
        if state_norm > 0:
            normalized_state = initial_state / state_norm
        else:
            normalized_state = torch.zeros_like(initial_state)
            
        # Register attractor
        self.attractors[attractor_id] = {
            "state": normalized_state,
            "importance": importance,
            "creation_time": self.update_count,
            "last_update": self.update_count,
            "stability": 1.0,
            "reinforcement_count": 0
        }
        
        # Initialize stability history
        self.stability_history[attractor_id] = [1.0]
        
        # Initialize metadata
        self.stabilization_metadata[attractor_id] = {
            "reinforcement_history": [],
            "alignment_history": []
        }
        
        logger.info(f"Registered attractor {attractor_id} with importance {importance}")
    
    def update_attractor(self,
                        attractor_id: str,
                        new_state: torch.Tensor,
                        learning_rate: float = 0.1) -> None:
        """
        Update an existing attractor with new state information.
        
        Parameters:
            attractor_id (str): Identifier of the attractor to update
            new_state (torch.Tensor): New state vector to incorporate
            learning_rate (float): Rate at which to update the attractor (0-1)
        """
        if attractor_id not in self.attractors:
            logger.warning(f"Attractor {attractor_id} does not exist. Creating new.")
            self.register_attractor(attractor_id, new_state)
            return
        
        # Normalize new state
        state_norm = torch.norm(new_state)
        if state_norm > 0:
            normalized_new = new_state / state_norm
        else:
            normalized_new = torch.zeros_like(new_state)
            
        # Get current state
        current_state = self.attractors[attractor_id]["state"]
        
        # Update state with exponential moving average
        updated_state = (1 - learning_rate) * current_state + learning_rate * normalized_new
        
        # Normalize updated state
        updated_norm = torch.norm(updated_state)
        if updated_norm > 0:
            updated_state = updated_state / updated_norm
            
        # Update attractor
        self.attractors[attractor_id]["state"] = updated_state
        self.attractors[attractor_id]["last_update"] = self.update_count
        
        logger.debug(f"Updated attractor {attractor_id} with learning rate {learning_rate}")
    
    def measure_attractor_alignment(self,
                                   attractor_id: str,
                                   current_state: torch.Tensor) -> float:
        """
        Measure alignment between current state and attractor.
        
        Parameters:
            attractor_id (str): Identifier of the attractor
            current_state (torch.Tensor): Current state to compare
            
        Returns:
            float: Alignment score (0-1)
        """
        if attractor_id not in self.attractors:
            return 0.0
            
        # Normalize current state
        state_norm = torch.norm(current_state)
        if state_norm > 0:
            normalized_current = current_state / state_norm
        else:
            return 0.0
            
        # Get attractor state
        attractor_state = self.attractors[attractor_id]["state"]
        
        # Calculate cosine similarity
        alignment = torch.cosine_similarity(
            normalized_current.unsqueeze(0),
            attractor_state.unsqueeze(0)
        ).item()
        
        # Scale from [-1, 1] to [0, 1]
        alignment = (alignment + 1) / 2
        
        return alignment
    
    def stabilize(self,
                 output_states: torch.Tensor,
                 attractor_strength: float,
                 phase_vector: torch.Tensor,
                 residue: torch.Tensor) -> torch.Tensor:
        """
        Stabilize output states based on attractor patterns.
        
        Parameters:
            output_states (torch.Tensor): Output hidden states to stabilize
            attractor_strength (float): Current attractor activation strength
            phase_vector (torch.Tensor): Current phase vector
            residue (torch.Tensor): Current symbolic residue
            
        Returns:
            torch.Tensor: Stabilized hidden states
        """
        self.update_count += 1
        batch_size, seq_len, hidden_dim = output_states.shape
        
        # Check if stabilization is needed
        if attractor_strength >= self.stability_warning:
            # No need for stabilization
            return output_states
            
        logger.info(f"Stabilizing with attractor strength {attractor_strength:.4f}")
        
        # Calculate adaptive reinforcement strength
        # Stronger reinforcement for lower attractor strength
        if attractor_strength < self.stability_critical:
            # Critical stabilization needed
            reinforcement_strength = self.max_reinforcement
        else:
            # Scale reinforcement based on how far below warning threshold
            normalized_strength = (attractor_strength - self.stability_critical) / (
                self.stability_warning - self.stability_critical
            )
            reinforcement_strength = self.min_reinforcement + (
                (self.max_reinforcement - self.min_reinforcement) * (1 - normalized_strength)
            )
            
        # If we don't have registered attractors yet, use phase vector as attractor
        if not self.attractors:
            attractor_id = "auto_phase_attractor"
            avg_phase = phase_vector.clone()
            
            # Register the phase vector as an attractor
            if attractor_id not in self.attractors:
                self.register_attractor(attractor_id, avg_phase)
            else:
                # Update existing attractor
                self.update_attractor(attractor_id, avg_phase, learning_rate=0.2)
        
        # Calculate stabilization influence from each attractor
        stabilized_states = output_states.clone()
        total_influence = 0.0
        
        for attractor_id, attractor in self.attractors.items():
            # Skip inactive attractors
            if self.update_count - attractor["last_update"] > self.config.get("attractor_timeout", 1000):
                continue
                
            # Calculate alignment with this attractor
            avg_output = output_states.mean(dim=(0, 1))
            alignment = self.measure_attractor_alignment(attractor_id, avg_output)
            
            # Store alignment in metadata
            self.stabilization_metadata[attractor_id]["alignment_history"].append(alignment)
            
            # Calculate influence based on alignment and importance
            influence = attractor["importance"] * (1.0 - alignment)
            total_influence += influence
            
            # Apply attractor influence to output
            attractor_state = attractor["state"].unsqueeze(0).unsqueeze(0)
            attractor_influence = attractor_state.expand_as(output_states)
            
            stabilized_states = stabilized_states + (
                reinforcement_strength * influence * attractor_influence
            )
            
            # Record reinforcement
            self.stabilization_metadata[attractor_id]["reinforcement_history"].append({
                "update": self.update_count,
                "alignment": alignment,
                "influence": influence,
                "reinforcement_strength": reinforcement_strength
            })
            
            # Update attractor stability
            stability = alignment * attractor_strength
            self.attractors[attractor_id]["stability"] = stability
            self.stability_history[attractor_id].append(stability)
            
            # Track reinforcement count
            self.attractors[attractor_id]["reinforcement_count"] += 1
            
            logger.debug(f"Applied {attractor_id} with alignment {alignment:.4f}, "
                         f"influence {influence:.4f}, strength {reinforcement_strength:.4f}")
        
        # Normalize stabilized states if needed
        if total_influence > 0:
            # Normalize each hidden state
            for b in range(batch_size):
                for s in range(seq_len):
                    state_norm = torch.norm(stabilized_states[b, s])
                    output_norm = torch.norm(output_states[b, s])
                    
                    if state_norm > 0 and output_norm > 0:
                        # Preserve original norm
                        stabilized_states[b, s] = stabilized_states[b, s] * (output_norm / state_norm)
            
            logger.info(f"Stabilization applied with total influence {total_influence:.4f}")
        
        return stabilized_states
    
    def detect_attractor_collapse(self, attractor_id: str, threshold: float = 0.3) -> bool:
        """
        Detect if an attractor is collapsing below stability threshold.
        
        Parameters:
            attractor_id (str): Identifier of the attractor
            threshold (float): Stability threshold
            
        Returns:
            bool: True if attractor is collapsing, False otherwise
        """
        if attractor_id not in self.attractors:
            return False
            
        stability = self.attractors[attractor_id]["stability"]
        
        return stability < threshold
    
    def get_attractor_stability(self, attractor_id: Optional[str] = None) -> Union[float, Dict[str, float]]:
        """
        Get the stability of a specific attractor or all attractors.
        
        Parameters:
            attractor_id (str, optional): Identifier of the attractor, or None for all
            
        Returns:
            float or Dict[str, float]: Stability value(s)
        """
        if attractor_id is not None:
            if attractor_id not in self.attractors:
                return 0.0
                
            return self.attractors[attractor_id]["stability"]
        else:
            return {
                attractor_id: attractor["stability"]
                for attractor_id, attractor in self.attractors.items()
            }
    
    def get_stability_trajectory(self, attractor_id: str, window_size: int = 10) -> List[float]:
        """
        Get recent stability trajectory for an attractor.
        
        Parameters:
            attractor_id (str): Attractor identifier
            window_size (int): Number of recent updates to consider
            
        Returns:
            List[float]: Recent stability values
        """
        if attractor_id not in self.stability_history:
            return [1.0]
            
        history = self.stability_history[attractor_id]
        
        # Get recent history up to window_size
        recent_history = history[-window_size:] if len(history) >= window_size else history
        
        return recent_history
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about the attractor system."""
        # Count active attractors
        active_count = sum(
            1 for attractor_id, attractor in self.attractors.items()
            if self.update_count - attractor["last_update"] <= self.config.get("attractor_timeout", 1000)
        )
        
        # Calculate average stability across active attractors
        active_stabilities = [
            attractor["stability"]
            for attractor_id, attractor in self.attractors.items()
            if self.update_count - attractor["last_update"] <= self.config.get("attractor_timeout", 1000)
        ]
        
        avg_stability = sum(active_stabilities) / max(1, len(active_stabilities))
        
        # Identify most and least stable attractors
        if self.attractors:
            most_stable_id = max(self.attractors, key=lambda x: self.attractors[x]["stability"])
            least_stable_id = min(self.attractors, key=lambda x: self.attractors[x]["stability"])
            
            most_stable = {
                "id": most_stable_id,
                "stability": self.attractors[most_stable_id]["stability"],
                "reinforcement_count": self.attractors[most_stable_id]["reinforcement_count"]
            }
            
            least_stable = {
                "id": least_stable_id,
                "stability": self.attractors[least_stable_id]["stability"],
                "reinforcement_count": self.attractors[least_stable_id]["reinforcement_count"]
            }
        else:
            most_stable = {"id": None, "stability": 1.0, "reinforcement_count": 0}
            least_stable = {"id": None, "stability": 1.0, "reinforcement_count": 0}
        
        # Attractor statistics
        attractor_stats = {}
        for attractor_id, attractor in self.attractors.items():
            recent_stability = self.get_stability_trajectory(attractor_id, window_size=5)
            
            attractor_stats[attractor_id] = {
                "current_stability": attractor["stability"],
                "importance": attractor["importance"],
                "age": self.update_count - attractor["creation_time"],
                "last_update_age": self.update_count - attractor["last_update"],
                "reinforcement_count": attractor["reinforcement_count"],
                "recent_stability": recent_stability,
                "is_active": self.update_count - attractor["last_update"] <= self.config.get("attractor_timeout", 1000)
            }
        
        return {
            "attractor_count": len(self.attractors),
            "active_count": active_count,
            "average_stability": avg_stability,
            "most_stable_attractor": most_stable,
            "least_stable_attractor": least_stable,
            "attractor_statistics": attractor_stats,
            "update_count": self.update_count
        }
    
    def reset(self) -> None:
        """Reset the attractor stabilization system state."""
        self.attractors = {}
        self.stability_history = {}
        self.stabilization_metadata = {}
        self.update_count = 0
        
        logger.info("Attractor Stabilization System reset")
