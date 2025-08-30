#!/usr/bin/env python3
"""
data_loader.py - Load and parse SHROOM datasets
Handles JSON array, JSONL, or dict-wrapped formats.
"""

import json
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path


class SHROOMLoader:
    """Load and parse SHROOM hallucination detection datasets"""

    def __init__(self, data_dir: str = "data/"):
        """Initialize with data directory path"""
        self.data_dir = Path(data_dir)
        self.schema_info: Dict[str, Any] = {}
        self._inspect_files()

    # ---------- helpers ----------

    def _determine_split(self, filename: str) -> Optional[str]:
        """Determine dataset split from filename"""
        filename_lower = filename.lower()
        if "train" in filename_lower:
            return "train"
        if "dev" in filename_lower or "val" in filename_lower:
            return "dev"
        if "test" in filename_lower:
            return "test"
        return None

    def _inspect_files(self) -> None:
        """Inspect available dataset files and their schemas"""
        if not self.data_dir.exists():
            print(f"Warning: Data directory {self.data_dir} does not exist")
            return

        json_files = list(self.data_dir.rglob("*.json"))
        self.schema_info = {
            "files_found": len(json_files),
            "file_paths": [str(f.relative_to(self.data_dir)) for f in json_files],
            "splits": {},
        }

        for file_path in json_files:
            split_name = self._determine_split(file_path.name)
            if split_name:
                self.schema_info["splits"][split_name] = str(file_path)

    def _read_examples(self, file_path: Path) -> List[Dict[str, Any]]:
        """
        Read a dataset file and always return a list of dict examples.

        Supports:
          - JSON array at root:        [ {...}, {...} ]
          - JSON dict with list field: { "examples": [ ... ] } / "data"/"items"/"records"/"instances"
          - JSONL (one object / line)
        """
        # Try whole-file JSON first
        text = file_path.read_text(encoding="utf-8").strip()
        if not text:
            return []

        def _as_examples(obj: Any) -> List[Dict[str, Any]]:
            # Normalize different shapes to List[Dict]
            if isinstance(obj, list):
                return [x for x in obj if isinstance(x, dict)]
            if isinstance(obj, dict):
                for key in ("examples", "data", "items", "records", "instances"):
                    if key in obj and isinstance(obj[key], list):
                        return [x for x in obj[key] if isinstance(x, dict)]
                # Single example dict
                return [obj]
            return []

        try:
            loaded = json.loads(text)
            examples = _as_examples(loaded)
            if examples:
                return examples
        except json.JSONDecodeError:
            # Fall back to JSONL parsing
            pass

        # JSONL fallback
        examples: List[Dict[str, Any]] = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    examples.append(obj)
                elif isinstance(obj, list):
                    examples.extend([x for x in obj if isinstance(x, dict)])
            except json.JSONDecodeError:
                # skip bad lines rather than crash
                continue
        return examples

    @staticmethod
    def _get_label(item: Dict[str, Any]) -> Any:
        for k in ("label", "labels", "gold_label", "hallucination", "is_hallucination"):
            if k in item:
                v = item[k]
                # normalize list labels like ["Hallucination"]
                if isinstance(v, list) and v:
                    return v[0]
                return v
        return "Unknown"

    @staticmethod
    def _is_hallucination(label: Any) -> bool:
        if isinstance(label, str):
            return label.strip().lower().startswith("halluc")
        if isinstance(label, (int, float, bool)):
            return bool(label)
        return False

    @staticmethod
    def _get_hyp(item: Dict[str, Any]) -> str:
        for k in ("hyp", "hypothesis", "output", "response", "generation", "pred"):
            if k in item and isinstance(item[k], str):
                return item[k]
        return ""

    # ---------- public API ----------

    def load_dataset(self, split: str = "train", model_type: str = "model-agnostic") -> List[Dict[str, Any]]:
        """
        Load SHROOM dataset split.

        Args:
            split: 'train' | 'dev' | 'test'
            model_type: 'model-agnostic' | 'model-aware'
        """
        matching_files: List[Path] = []
        for file_path in self.data_dir.rglob("*.json"):
            filename = file_path.name.lower()
            if split in filename and model_type.replace("-", "") in filename.replace("-", ""):
                matching_files.append(file_path)

        if not matching_files:
            raise FileNotFoundError(f"No {split} file found for {model_type} in {self.data_dir}")

        file_path = matching_files[0]
        print(f"Loading {split} dataset from: {file_path.relative_to(self.data_dir)}")

        data = self._read_examples(file_path)
        print(f"Loaded {len(data)} examples from {split} split")
        return data

    def get_dataset_info(self, split: str = "train", model_type: str = "model-agnostic") -> Dict[str, Any]:
        """Get basic statistics about the dataset"""
        try:
            data = self.load_dataset(split, model_type)

            fields = sorted({k for item in data for k in item.keys()}) if data else []

            labels_list: List[Any] = [self._get_label(item) for item in data]
            label_counts: Dict[str, int] = {}
            for lab in labels_list:
                key = str(lab)
                label_counts[key] = label_counts.get(key, 0) + 1

            halluc_count = sum(1 for lab in labels_list if self._is_hallucination(lab))

            # hypothesis lengths (best-effort)
            hyp_lengths = [len(self._get_hyp(item).split()) for item in data]
            hyp_lengths = [n for n in hyp_lengths if n > 0]

            info: Dict[str, Any] = {
                "total_examples": len(data),
                "fields": fields,
                "tasks": {t: 0 for t in set(item.get("task", "Unknown") for item in data)},
                "labels": label_counts,
                "hallucination_rate": (halluc_count / len(data)) if data else 0.0,
            }

            # populate task counts
            for item in data:
                t = item.get("task", "Unknown")
                info["tasks"][t] = info["tasks"].get(t, 0) + 1

            if hyp_lengths:
                info["avg_hyp_length"] = sum(hyp_lengths) / len(hyp_lengths)
                info["max_hyp_length"] = max(hyp_lengths)
                info["min_hyp_length"] = min(hyp_lengths)

            return info

        except Exception as e:
            return {"error": str(e), "total_examples": 0}

    def extract_hypothesis_texts(self, split: str = "train", model_type: str = "model-agnostic") -> List[Tuple[str, str]]:
        """
        Extract hypothesis texts with their labels for pipeline processing.

        Returns:
            List of (hypothesis_text, label) tuples
        """
        data = self.load_dataset(split, model_type)

        out: List[Tuple[str, str]] = []
        for item in data:
            hyp = self._get_hyp(item).strip()
            lab = self._get_label(item)
            lab_str = str(lab) if lab is not None else "Unknown"
            if hyp:
                out.append((hyp, lab_str))
        return out

    def get_available_splits(self) -> Dict[str, List[str]]:
        """Get all available dataset splits and model types"""
        available = {"splits": [], "model_types": []}

        for file_path in self.data_dir.rglob("*.json"):
            filename = file_path.name.lower()

            # Determine split
            if "train" in filename:
                if "train" not in available["splits"]:
                    available["splits"].append("train")
            elif "dev" in filename or "val" in filename:
                if "dev" not in available["splits"]:
                    available["splits"].append("dev")
            elif "test" in filename:
                if "test" not in available["splits"]:
                    available["splits"].append("test")

            # Determine model type
            if "agnostic" in filename:
                if "model-agnostic" not in available["model_types"]:
                    available["model_types"].append("model-agnostic")
            elif "aware" in filename:
                if "model-aware" not in available["model_types"]:
                    available["model_types"].append("model-aware")

        return available


def demo():
    """Demo the SHROOM data loader"""
    loader = SHROOMLoader()

    print("SHROOM Dataset Loader Demo")
    print("=" * 40)

    # Show available datasets
    available = loader.get_available_splits()
    print(f"Available splits: {available['splits']}")
    print(f"Available model types: {available['model_types']}")

    # Try to load a dataset
    try:
        if available["splits"]:
            split = available["splits"][0]
            model_type = available["model_types"][0] if available["model_types"] else "model-agnostic"

            print(f"\nLoading {split} dataset ({model_type})...")
            info = loader.get_dataset_info(split, model_type)
            print(f"Dataset info: {info}")

            # Sample some texts
            texts_labels = loader.extract_hypothesis_texts(split, model_type)
            if texts_labels:
                print(f"\nFirst 3 hypothesis examples:")
                for i, (text, label) in enumerate(texts_labels[:3], 1):
                    print(f"{i}. Label: {label}")
                    print(f"   Text: {text[:100]}...\n")

    except Exception as e:
        print(f"Error loading dataset: {e}")


if __name__ == "__main__":
    demo()