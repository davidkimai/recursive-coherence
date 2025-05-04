# recursive_entropy_manager/core/metabolism.py

import torch
import torch.nn as nn
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Union, Any

logger = logging.getLogger(__name__)


class ContradictionMetabolismEngine:
    """
    Implements the Contradiction Metabolism Engine for the Recursive Entropy Manager.
    
    This engine enables transformers to process and integrate contradictions while
    maintaining coherence. It determines how and when contradictions are metabolized
    based on the system's current state.
    
    Key functionality:
    - Detects contradictions in input and internal states
    - Measures the system's capacity to metabolize contradictions
    - Processes contradictions without destabilizing the system
    - Adaptively adjusts metabolism rates based on current coherence
    
    Mathematical foundation:
    - Uses the Elastic Tolerance (λp() to determine contradiction processing capacity
    - Leverages Phase Alignment (τ(p,t)) to guide contradiction integration
    - Considers Bounded Integrity (B(p)) to maintain system boundaries
    """
    
    def __init__(self, hidden_dim: int, config: Dict[str, Any]):
        """
        Initialize the Contradiction Metabolism Engine.
        
        Parameters:
            hidden_dim (int): Dimension of the hidden states
            config (Dict): Configuration parameters
        """
        self.hidden_dim = hidden_dim
        self.config = config
        
        # Contradiction registry
        self.contradiction_store = {}
        
        # Metabolism statistics
        self.metabolism_stats = {
            "processed_count": 0,
            "deferred_count": 0,
            "rejected_count": 0,
            "processing_cost": [],
            "metabolized_contradictions": []
        }
        
        # Metabolism thresholds
        self.processing_threshold = config.get("processing_threshold", 0.3)
        self.cost_warning = config.get("cost_warning", 0.7)
        
        # Internal state
        self.update_count = 0
        self.layer_state = {}
        
        logger.info("Contradiction Metabolism Engine initialized")
    
    def register_contradiction(self, 
                              contradiction_id: str, 
                              content: torch.Tensor,
                              source: str,
                              priority: float = 0.5) -> None:
        """
        Register a new contradiction to be processed.
        
        Parameters:
            contradiction_id (str): Unique identifier for the contradiction
            content (torch.Tensor): Vector representation of the contradiction
            source (str): Source of the contradiction
            priority (float): Processing priority (0-1)
        """
        if contradiction_id in self.contradiction_store:
            logger.warning(f"Contradiction {contradiction_id} already exists. Updating.")
        
        # Register contradiction
        self.contradiction_store[contradiction_id] = {
            "content": content,
            "source": source,
            "priority": priority,
            "registration_time": self.update_count,
            "processing_attempts": 0,
            "status": "pending",
            "cost": 0.0
        }
        
        logger.info(f"Registered contradiction {contradiction_id} from {source} with priority {priority}")
    
    def detect_contradictions(self,
                             states: torch.Tensor,
                             layer_idx: int) -> List[Dict[str, Any]]:
        """
        Detect contradictions in the hidden states.
        
        Parameters:
            states (torch.Tensor): Hidden states to analyze
            layer_idx (int): Layer index
            
        Returns:
            List[Dict]: Detected contradictions
        """
        # Initialize layer state if needed
        if layer_idx not in self.layer_state:
            self.layer_state[layer_idx] = {
                "previous_states": None,
                "contradiction_history": []
            }
        
        # No previous states available for comparison
        if self.layer_state[layer_idx]["previous_states"] is None:
            self.layer_state[layer_idx]["previous_states"] = states.detach()
            return []
        
        # Get previous states
        previous_states = self.layer_state[layer_idx]["previous_states"]
        
        # Detect contradictions using a simplified approach:
        # Look for significant directional shifts in the hidden state
        batch_size, seq_len, hidden_dim = states.shape
        
        # Average across batch dimension for simplicity
        avg_current = states.mean(dim=0)
        avg_previous = previous_states.mean(dim=0)
        
        # Identify token positions with significant directional changes
        contradictions = []
        
        for i in range(seq_len):
            current_vector = avg_current[i]
            previous_vector = avg_previous[i] if i < avg_previous.shape[0] else torch.zeros_like(current_vector)
            
            # Calculate change vector
            change_vector = current_vector - previous_vector
            change_magnitude = torch.norm(change_vector).item()
            
            # Check if change is significant enough to represent a contradiction
            if change_magnitude > self.config.get("contradiction_threshold", 0.5):
                # Calculate directional similarity to see if it's a shift or an extension
                if torch.norm(previous_vector) > 1e-6:
                    direction_similarity = torch.cosine_similarity(
                        previous_vector.unsqueeze(0),
                        current_vector.unsqueeze(0)
                    ).item()
                    
                    # Contradiction is detected if directions are significantly different
                    # (i.e., low or negative similarity)
                    if direction_similarity < self.config.get("direction_threshold", 0.5):
                        contradiction_id = f"layer{layer_idx}_pos{i}_t{self.update_count}"
                        
                        # Create contradiction record
                        contradiction = {
                            "id": contradiction_id,
                            "position": i,
                            "magnitude": change_magnitude,
                            "direction_similarity": direction_similarity,
                            "content": change_vector,
                            "priority": (1.0 - max(0, direction_similarity)) * min(1.0, change_magnitude),
                            "source": f"layer{layer_idx}"
                        }
                        
                        contradictions.append(contradiction)
                        
                        # Register for processing
                        self.register_contradiction(
                            contradiction_id=contradiction_id,
                            content=change_vector,
                            source=f"layer{layer_idx}",
                            priority=contradiction["priority"]
                        )
        
        # Update previous states
        self.layer_state[layer_idx]["previous_states"] = states.detach()
        
        # Store contradiction history
        self.layer_state[layer_idx]["contradiction_history"].append(len(contradictions))
        
        logger.debug(f"Detected {len(contradictions)} contradictions in layer {layer_idx}")
        
        return contradictions
    
    def calculate_metabolism_capacity(self,
                                     coherence: float,
                                     phase_alignment: float) -> float:
        """
        Calculate current capacity to metabolize contradictions.
        
        Parameters:
            coherence (float): Current coherence value
            phase_alignment (float): Current phase alignment
            
        Returns:
            float: Metabolism capacity (0-1)
        """
        # Metabolism capacity is a function of coherence and phase alignment
        # Higher coherence and alignment = higher capacity
        
        # Base capacity from coherence
        base_capacity = coherence
        
        # Phase alignment factor
        # Higher alignment = better capacity
        alignment_factor = phase_alignment
        
        # Combine factors
        capacity = base_capacity * alignment_factor
        
        # Apply nonlinear scaling to accentuate high and low values
        # This creates clearer thresholds for metabolism decisions
        scaled_capacity = capacity ** self.config.get("capacity_exponent", 1.5)
        
        return scaled_capacity
    
    def calculate_processing_cost(self,
                                 contradiction: Dict[str, Any],
                                 coherence: float,
                                 phase_alignment: float) -> float:
        """
        Calculate the cost of processing a contradiction.
        
        Parameters:
            contradiction (Dict): Contradiction to process
            coherence (float): Current coherence value
            phase_alignment (float): Current phase alignment
            
        Returns:
            float: Processing cost (0-1)
        """
        # Contradiction properties
        priority = contradiction["priority"]
        magnitude = contradiction.get("magnitude", torch.norm(contradiction["content"]).item())
        
        # Contradiction size factor
        # Larger contradictions cost more to process
        size_factor = min(1.0, magnitude / self.config.get("magnitude_normalizer", 2.0))
        
        # Priority factor
        # Higher priority contradictions may cost more, as they represent more significant issues
        priority_factor = priority
        
        # Phase misalignment factor
        # Lower alignment = higher cost
        misalignment_factor = 2.0 - phase_alignment
        
        # Coherence factor
        # Lower coherence = higher cost
        coherence_factor = 2.0 - coherence
        
        # Combine factors with weighted importance
        # Customize weights based on specific model requirements
        cost = (
            self.config.get("size_weight", 0.3) * size_factor +
            self.config.get("priority_weight", 0.2) * priority_factor +
            self.config.get("misalignment_weight", 0.3) * misalignment_factor +
            self.config.get("coherence_weight", 0.2) * coherence_factor
        )
        
        # Normalize to [0, 1] range
        normalized_cost = min(1.0, max(0.0, cost / 2.0))
        
        return normalized_cost
    
    def prioritize_contradictions(self, 
                                 contradictions: List[Dict[str, Any]],
                                 capacity: float) -> List[Dict[str, Any]]:
        """
        Prioritize contradictions for processing based on available capacity.
        
        Parameters:
            contradictions (List[Dict]): Contradictions to prioritize
            capacity (float): Current metabolism capacity
            
        Returns:
            List[Dict]: Prioritized contradictions to process
        """
        # If no contradictions, return empty list
        if not contradictions:
            return []
            
        # Sort contradictions by priority (descending)
        sorted_contradictions = sorted(
            contradictions,
            key=lambda x: x["priority"],
            reverse=True
        )
        
        # Calculate how many contradictions can be processed
        # Higher capacity = more contradictions
        max_contradictions = max(1, int(len(sorted_contradictions) * capacity))
        
        # Get highest priority contradictions up to capacity
        prioritized = sorted_contradictions[:max_contradictions]
        
        return prioritized
    
    def metabolize_contradiction(self,
                                contradiction: Dict[str, Any],
                                states: torch.Tensor,
                                coherence: float,
                                phase_alignment: float) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Metabolize a single contradiction.
        
        Parameters:
            contradiction (Dict): Contradiction to metabolize
            states (torch.Tensor): States to update
            coherence (float): Current coherence value
            phase_alignment (float): Current phase alignment
            
        Returns:
            Tuple[torch.Tensor, Dict]: (Updated states, Metabolism stats)
        """
        # Calculate processing cost
        cost = self.calculate_processing_cost(contradiction, coherence, phase_alignment)
        
        # Update contradiction record
        if contradiction["id"] in self.contradiction_store:
            self.contradiction_store[contradiction["id"]]["processing_attempts"] += 1
            self.contradiction_store[contradiction["id"]]["cost"] = cost
        
        # Check if cost is too high for current capacity
        capacity = self.calculate_metabolism_capacity(coherence, phase_alignment)
        
        if cost > capacity:
            # Too expensive to process now, defer for later
            if contradiction["id"] in self.contradiction_store:
                self.contradiction_store[contradiction["id"]]["status"] = "deferred"
            
            self.metabolism_stats["deferred_count"] += 1
            
            logger.debug(f"Contradiction {contradiction['id']} deferred: "
                        f"cost {cost:.4f} > capacity {capacity:.4f}")
            
            # Return states unchanged
            return states, {
                "id": contradiction["id"],
                "action": "deferred",
                "cost": cost,
                "capacity": capacity,
                "reason": "cost_exceeds_capacity"
            }
        
        # Process the contradiction
        # This is a simplified approach - in a real implementation, this would
        # be more sophisticated and tailored to the specific model architecture
        
        batch_size, seq_len, hidden_dim = states.shape
        contradiction_content = contradiction["content"]
        
        # Calculate integration factor based on coherence and phase alignment
        # Higher coherence and alignment = more integration
        integration_factor = capacity * (1.0 - cost)
        
        # Scale to appropriate range
        integration_factor *= self.config.get("integration_scale", 0.3)
        
        # Apply contradiction to states
        if "position" in contradiction:
            # Apply to specific position if known
            position = contradiction["position"]
            if position < seq_len:
                # Expand contradiction to batch dimension
                expanded_content = contradiction_content.unsqueeze(0).expand(batch_size, -1)
                
                # Apply integration
                states[:, position] += integration_factor * expanded_content
        else:
            # Apply globally if no specific position
            # Calculate global integration based on cosine similarity
            for i in range(seq_len):
                for b in range(batch_size):
                    # Calculate similarity to current token
                    similarity = torch.cosine_similarity(
                        states[b, i].unsqueeze(0),
                        contradiction_content.unsqueeze(0)
                    ).item()
                    
                    # Scale integration by similarity
                    # More similar tokens get more effect
                    token_integration = integration_factor * abs(similarity)
                    
                    # Apply integrated change
                    # Use sign of similarity to determine direction
                    direction = 1.0 if similarity >= 0 else -1.0
                    states[b, i] += direction * token_integration * contradiction_content
        
        # Update contradiction record
        if contradiction["id"] in self.contradiction_store:
            self.contradiction_store[contradiction["id"]]["status"] = "processed"
        
        self.metabolism_stats["processed_count"] += 1
        self.metabolism_stats["processing_cost"].append(cost)
        self.metabolism_stats["metabolized_contradictions"].append(contradiction["id"])
        
        # Return updated states and stats
        return states, {
            "id": contradiction["id"],
            "action": "processed",
            "cost": cost,
            "capacity": capacity,
            "integration_factor": integration_factor
        }
    
    def metabolize(self,
                  states: torch.Tensor,
                  coherence: float,
                  phase_alignment: float,
                  residue: torch.Tensor,
                  layer_idx: int) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Metabolize contradictions in the current states.
        
        This is the main entry point for the contradiction metabolism process.
        
        Parameters:
            states (torch.Tensor): States to process
            coherence (float): Current coherence value
            phase_alignment (float): Current phase alignment
            residue (torch.Tensor): Current symbolic residue
            layer_idx (int): Layer index
            
        Returns:
            Tuple[torch.Tensor, Dict]: (Updated states, Metabolism stats)
        """
        self.update_count += 1
        
        # First, detect contradictions in the current states
        detected_contradictions = self.detect_contradictions(states, layer_idx)
        
        # Calculate current metabolism capacity
        capacity = self.calculate_metabolism_capacity(coherence, phase_alignment)
        
        # If capacity is too low, skip metabolism
        if capacity < self.processing_threshold:
            logger.info(f"Metabolism capacity ({capacity:.4f}) below threshold "
                       f"({self.processing_threshold}). Skipping metabolism.")
            
            return states, {
                "action": "skipped",
                "capacity": capacity,
                "threshold": self.processing_threshold,
                "detected_count": len(detected_contradictions),
                "processed_count": 0
            }
        
        # Prioritize contradictions to process
        process_queue = self.prioritize_contradictions(detected_contradictions, capacity)
        
        # Process contradictions
        updated_states = states.clone()
        processed_results = []
        
        for contradiction in process_queue:
            # Metabolize contradiction
            updated_states, result = self.metabolize_contradiction(
                contradiction, updated_states, coherence, phase_alignment
            )
            
            processed_results.append(result)
        
        # Return updated states and stats
        return updated_states, {
            "action": "metabolized",
            "capacity": capacity,
            "detected_count": len(detected_contradictions),
            "processed_count": len(processed_results),
            "results": processed_results
        }
    
    def get_contradiction_status(self, contradiction_id: str) -> Dict[str, Any]:
        """
        Get the status of a specific contradiction.
        
        Parameters:
            contradiction_id (str): Contradiction identifier
            
        Returns:
            Dict: Contradiction status
        """
        if contradiction_id not in self.contradiction_store:
            return {"status": "unknown"}
            
        return self.contradiction_store[contradiction_id]
    
    def get_layer_contradiction_history(self, layer_idx: int) -> List[int]:
        """
        Get contradiction history for a specific layer.
        
        Parameters:
            layer_idx (int): Layer index
            
        Returns:
            List[int]: Number of contradictions detected per update
        """
        if layer_idx not in self.layer_state:
            return []
            
        return self.layer_state[layer_idx]["contradiction_history"]
    
    def get_average_processing_cost(self, window_size: int = 10) -> float:
        """
        Get the average processing cost over recent updates.
        
        Parameters:
            window_size (int): Number of recent updates to consider
            
        Returns:
            float: Average processing cost
        """
        costs = self.metabolism_stats["processing_cost"]
        
        if not costs:
            return 0.0
            
        # Get recent costs up to window_size
        recent_costs = costs[-window_size:] if len(costs) >= window_size else costs
        
        return sum(recent_costs) / len(recent_costs)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about the metabolism engine."""
        # Count contradictions by status
        status_counts = {
            "pending": 0,
            "processed": 0,
            "deferred": 0,
            "rejected": 0
        }
        
        for contradiction_id, contradiction in self.contradiction_store.items():
            status = contradiction["status"]
            if status in status_counts:
                status_counts[status] += 1
            else:
                status_counts[status] = 1
        
        # Calculate average processing cost
        avg_cost = self.get_average_processing_cost()
        
        # Gather layer-specific stats
        layer_stats = {}
        for layer_idx, state in self.layer_state.items():
            history = state["contradiction_history"]
            
            layer_stats[layer_idx] = {
                "total_contradictions": sum(history),
                "recent_contradictions": history[-10:] if len(history) >= 10 else history,
                "average_per_update": sum(history) / max(1, len(history))
            }
        
        # Recent metabolism rate
        recent_processed = len(self.metabolism_stats["metabolized_contradictions"][-20:]) if self.metabolism_stats["metabolized_contradictions"] else 0
        recent_deferred = min(20, self.metabolism_stats["deferred_count"])
        
        if recent_processed + recent_deferred > 0:
            recent_metabolism_rate = recent_processed / (recent_processed + recent_deferred)
        else:
            recent_metabolism_rate = 0.0
        
        return {
            "contradiction_count": len(self.contradiction_store),
            "status_counts": status_counts,
            "processed_count": self.metabolism_stats["processed_count"],
            "deferred_count": self.metabolism_stats["deferred_count"],
            "rejected_count": self.metabolism_stats["rejected_count"],
            "average_processing_cost": avg_cost,
            "recent_metabolism_rate": recent_metabolism_rate,
            "layer_statistics": layer_stats,
            "update_count": self.update_count
        }
    
    def reset(self) -> None:
        """Reset the contradiction metabolism engine state."""
        self.contradiction_store = {}
        self.metabolism_stats = {
            "processed_count": 0,
            "deferred_count": 0,
            "rejected_count": 0,
            "processing_cost": [],
            "metabolized_contradictions": []
        }
        self.update_count = 0
        self.layer_state = {}
        
        logger.info("Contradiction Metabolism Engine reset")
