"""
Experiment Memory System - Tracks completed experiments to prevent duplicates
and enable automated workflow memory.
"""

import json
import os
import streamlit as st
from typing import Dict, Any, List, Optional
from datetime import datetime
from pathlib import Path


class ExperimentMemory:
    """Tracks completed experiments and their results."""
    
    def __init__(self, memory_file: str = "experiment_memory.json"):
        self.memory_file = memory_file
        self.memory_dir = "data"
        self._ensure_memory_dir()
    
    def _ensure_memory_dir(self):
        """Ensure memory directory exists."""
        if not os.path.exists(self.memory_dir):
            os.makedirs(self.memory_dir, exist_ok=True)
    
    def _load_memory(self) -> Dict[str, Any]:
        """Load experiment memory from file."""
        file_path = os.path.join(self.memory_dir, self.memory_file)
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                st.warning(f"Could not load experiment memory: {e}")
                return {"experiments": [], "metadata": {}}
        return {"experiments": [], "metadata": {}}
    
    def _save_memory(self, memory_data: Dict[str, Any]):
        """Save experiment memory to file."""
        file_path = os.path.join(self.memory_dir, self.memory_file)
        try:
            with open(file_path, 'w') as f:
                json.dump(memory_data, f, indent=2)
        except Exception as e:
            st.error(f"Could not save experiment memory: {e}")
    
    def add_experiment(
        self,
        experiment_id: str,
        description: str,
        composition: Optional[Dict[str, Any]] = None,
        results: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Add a completed experiment to memory.
        
        Returns True if experiment was added, False if it already exists.
        """
        memory_data = self._load_memory()
        
        # Check if experiment already exists
        existing_ids = [exp.get("experiment_id") for exp in memory_data.get("experiments", [])]
        if experiment_id in existing_ids:
            return False
        
        experiment_entry = {
            "experiment_id": experiment_id,
            "description": description,
            "composition": composition or {},
            "results": results or {},
            "metadata": metadata or {},
            "timestamp": datetime.now().isoformat(),
        }
        
        memory_data.setdefault("experiments", []).append(experiment_entry)
        
        # Update metadata
        memory_data["metadata"] = {
            "last_updated": datetime.now().isoformat(),
            "total_experiments": len(memory_data["experiments"]),
        }
        
        self._save_memory(memory_data)
        return True
    
    def has_experiment(self, experiment_id: str) -> bool:
        """Check if an experiment has already been completed."""
        memory_data = self._load_memory()
        existing_ids = [exp.get("experiment_id") for exp in memory_data.get("experiments", [])]
        return experiment_id in existing_ids
    
    def get_experiment(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        """Get details of a specific experiment."""
        memory_data = self._load_memory()
        for exp in memory_data.get("experiments", []):
            if exp.get("experiment_id") == experiment_id:
                return exp
        return None
    
    def get_all_experiments(self) -> List[Dict[str, Any]]:
        """Get all completed experiments."""
        memory_data = self._load_memory()
        return memory_data.get("experiments", [])
    
    def get_experiment_summary(self) -> str:
        """Get a summary of all completed experiments."""
        experiments = self.get_all_experiments()
        if not experiments:
            return "No experiments completed yet."
        
        summary_parts = [f"**Total Experiments Completed:** {len(experiments)}\n"]
        
        for i, exp in enumerate(experiments[-10:], 1):  # Last 10 experiments
            exp_id = exp.get("experiment_id", "Unknown")
            desc = exp.get("description", "No description")
            timestamp = exp.get("timestamp", "Unknown")
            summary_parts.append(f"{i}. **{exp_id}**: {desc[:100]}... ({timestamp[:10]})")
        
        return "\n".join(summary_parts)
    
    def find_similar_experiments(
        self,
        composition: Dict[str, Any],
        threshold: float = 0.9,
    ) -> List[Dict[str, Any]]:
        """
        Find experiments with similar compositions.
        Returns experiments with composition similarity >= threshold.
        """
        experiments = self.get_all_experiments()
        similar = []
        
        for exp in experiments:
            exp_comp = exp.get("composition", {})
            if not exp_comp:
                continue
            
            # Simple similarity check (can be enhanced)
            similarity = self._calculate_composition_similarity(composition, exp_comp)
            if similarity >= threshold:
                similar.append({**exp, "similarity": similarity})
        
        return sorted(similar, key=lambda x: x.get("similarity", 0), reverse=True)
    
    def _calculate_composition_similarity(
        self,
        comp1: Dict[str, Any],
        comp2: Dict[str, Any],
    ) -> float:
        """Calculate similarity between two compositions."""
        if not comp1 or not comp2:
            return 0.0
        
        # Get all keys
        all_keys = set(comp1.keys()) | set(comp2.keys())
        if not all_keys:
            return 0.0
        
        # Calculate similarity
        matches = 0
        total_diff = 0.0
        
        for key in all_keys:
            val1 = comp1.get(key, 0)
            val2 = comp2.get(key, 0)
            
            if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                diff = abs(val1 - val2)
                total_diff += diff
                if diff < 0.01:  # Very close values
                    matches += 1
            elif val1 == val2:
                matches += 1
        
        # Normalize similarity
        similarity = matches / len(all_keys) if all_keys else 0.0
        return similarity
    
    def clear_memory(self):
        """Clear all experiment memory."""
        memory_data = {"experiments": [], "metadata": {}}
        self._save_memory(memory_data)


def get_experiment_memory() -> ExperimentMemory:
    """Get or create experiment memory instance."""
    if "experiment_memory" not in st.session_state:
        memory_file = st.session_state.get("experiment_memory_file", "experiment_memory.json")
        st.session_state.experiment_memory = ExperimentMemory(memory_file=memory_file)
    return st.session_state.experiment_memory
