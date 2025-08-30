# tests/test_integration.py
import pytest
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_loader import SHROOMLoader
from pipeline import ClaimsPipeline
from evaluator import PipelineEvaluator
from benchmark import RuleAsDataNovelty

def test_data_loader_initialization():
    """Test SHROOM data loader can initialize"""
    loader = SHROOMLoader()
    assert loader is not None
    assert hasattr(loader, 'data_dir')

def test_pipeline_integration():
    """Test full pipeline integration"""
    pipeline = ClaimsPipeline()
    
    test_text = "Apple Inc. might see 25% growth, which seems unrealistic at 500% in one day."
    result = pipeline.process_text(test_text)
    
    assert 'sentences' in result
    assert 'summary' in result
    assert len(result['sentences']) > 0
    
    sentence = result['sentences'][0]
    assert 'spec_score' in sentence
    assert 'num_sanity_score' in sentence
    assert 'instability_score' in sentence
    assert 'flags' in sentence
    assert 'paraphrases' in sentence

def test_evaluator_initialization():
    """Test evaluator can initialize with components"""
    loader = SHROOMLoader()
    pipeline = ClaimsPipeline()
    evaluator = PipelineEvaluator(pipeline, loader)
    
    assert evaluator.pipeline is not None
    assert evaluator.data_loader is not None

def test_rule_as_data_novelty():
    """Test rule-as-data novelty lever"""
    pipeline = ClaimsPipeline()
    rule_as_data = RuleAsDataNovelty(pipeline)
    
    test_texts = [
        "The company definitely reported strong results.",
        "Revenue might increase by 15% this quarter.",
        "This seems like a realistic estimate."
    ]
    
    silver_labels = rule_as_data.generate_silver_labels(test_texts)
    assert len(silver_labels) == len(test_texts)
    
    for label in silver_labels:
        assert 'silver_speculative' in label
        assert 'silver_hallucination' in label
        assert 'spec_score' in label

def test_mock_evaluation():
    """Test evaluation with mock data"""
    loader = SHROOMLoader()
    pipeline = ClaimsPipeline()
    evaluator = PipelineEvaluator(pipeline, loader)
    
    # Mock some predictions and ground truth
    mock_predictions = [
        {'spec_score': 0.3, 'num_sanity_score': 0.1, 'instability_score': 0.2, 'total_flags': 0,
         'hedge_count': 2, 'absolute_count': 0, 'entity_count': 1, 'number_count': 1, 'percent_count': 1}
    ]
    mock_ground_truth = [1]  # Hallucination
    
    metrics = evaluator._calculate_metrics(mock_predictions, mock_ground_truth)
    assert 'feature_correlations' in metrics

if __name__ == "__main__":
    pytest.main([__file__, "-v"])