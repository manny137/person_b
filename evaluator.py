#!/usr/bin/env python3
"""
evaluator.py - Evaluate pipeline performance on SHROOM dataset
Calculate metrics and run ablation studies
"""

import json
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from sklearn.metrics import precision_recall_fscore_support, classification_report, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import os

from data_loader import SHROOMLoader
from pipeline import ClaimsPipeline


class PipelineEvaluator:
    """Evaluate claims pipeline against SHROOM hallucination dataset"""
    
    def __init__(self, pipeline: ClaimsPipeline, data_loader: SHROOMLoader):
        """Initialize with pipeline and data loader"""
        self.pipeline = pipeline
        self.data_loader = data_loader
        self.results_cache = {}
    
    def evaluate_on_split(self, split: str = "dev", model_type: str = "model-agnostic", 
                         max_samples: Optional[int] = None) -> Dict[str, Any]:
        """
        Evaluate pipeline on a dataset split
        
        Args:
            split: Dataset split to evaluate on
            model_type: SHROOM model type
            max_samples: Limit number of samples (for faster testing)
            
        Returns:
            Comprehensive evaluation results
        """
        # Allow "val" as alias for "dev"
        actual_split = split
        if split == "dev":
            try:
                # Try normal dev first
                _ = self.data_loader.extract_hypothesis_texts("dev", model_type)
            except FileNotFoundError:
                # Fallback to val
                print("⚠️ Dev split not found, falling back to val split...")
                actual_split = "val"

        print(f"Evaluating on {actual_split} split ({model_type})...")
        
        # Load dataset
        texts_labels = self.data_loader.extract_hypothesis_texts(actual_split, model_type)
        
        if max_samples:
            texts_labels = texts_labels[:max_samples]
            print(f"Limited to {max_samples} samples for testing")
        
        # Process texts through pipeline
        predictions = []
        ground_truth = []
        detailed_results = []
        
        for i, (text, label) in enumerate(texts_labels):
            if i % 100 == 0:
                print(f"Processed {i}/{len(texts_labels)} texts...")
            
            try:
                # Run pipeline
                result = self.pipeline.process_text(text)
                
                # Extract features for evaluation
                sentence_results = result.get('sentences', [])
                if sentence_results:
                    sent_result = sentence_results[0]  # Use first sentence
                    
                    prediction_features = {
                        'spec_score': sent_result.get('spec_score', 0.0),
                        'num_sanity_score': sent_result.get('num_sanity_score', 0.0),
                        'instability_score': sent_result.get('instability_score', 0.0),
                        'total_flags': len(sent_result.get('flags', [])),
                        'hedge_count': sent_result.get('spec_counts', {}).get('hedges', 0),
                        'absolute_count': sent_result.get('spec_counts', {}).get('absolutes', 0),
                        'entity_count': len(sent_result.get('claims', {}).get('entities', [])),
                        'number_count': len(sent_result.get('claims', {}).get('numbers', [])),
                        'percent_count': len(sent_result.get('claims', {}).get('percents', []))
                    }
                else:
                    # Empty result fallback
                    prediction_features = {k: 0.0 for k in [
                        'spec_score', 'num_sanity_score', 'instability_score',
                        'total_flags', 'hedge_count', 'absolute_count',
                        'entity_count', 'number_count', 'percent_count'
                    ]}
                
                predictions.append(prediction_features)
                ground_truth.append(1 if label == 'Hallucination' else 0)
                
                detailed_results.append({
                    'text': text,
                    'true_label': label,
                    'pipeline_result': result,
                    'features': prediction_features
                })
                
            except Exception as e:
                print(f"Error processing text {i}: {e}")
                # Add default values for failed processing
                predictions.append({k: 0.0 for k in [
                    'spec_score', 'num_sanity_score', 'instability_score',
                    'total_flags', 'hedge_count', 'absolute_count',
                    'entity_count', 'number_count', 'percent_count'
                ]})
                ground_truth.append(1 if label == 'Hallucination' else 0)
        
        # Calculate evaluation metrics
        evaluation_results = self._calculate_metrics(predictions, ground_truth)
        evaluation_results['detailed_results'] = detailed_results
        evaluation_results['split'] = actual_split
        evaluation_results['model_type'] = model_type
        evaluation_results['total_samples'] = len(texts_labels)
        
        return evaluation_results
    
    def _calculate_metrics(self, predictions: List[Dict], ground_truth: List[int]) -> Dict[str, Any]:
        """Calculate various evaluation metrics"""
        if not predictions:
            return {'error': 'No predictions to evaluate'}
        
        # Convert predictions to DataFrame for easier analysis
        pred_df = pd.DataFrame(predictions).fillna(0.0)
        
        results = {
            'feature_correlations': {},
            'threshold_analysis': {},
            'feature_importance': {}
        }
        
        # Analyze correlation between features and ground truth
        labels_have_variance = len(set(ground_truth)) > 1
        for feature in pred_df.columns:
            values = pred_df[feature].values.astype(float)
            feature_has_variance = np.nanstd(values) > 0
            if feature_has_variance and labels_have_variance:
                corr = np.corrcoef(values, ground_truth)
                if isinstance(corr, np.ndarray) and corr.shape == (2, 2) and np.isfinite(corr[0, 1]):
                    results['feature_correlations'][feature] = float(corr[0, 1])
                else:
                    results['feature_correlations'][feature] = 0.0
        
        # Threshold analysis
        key_features = ['spec_score', 'num_sanity_score', 'instability_score', 'total_flags']
        for feature in key_features:
            if feature in pred_df.columns:
                feature_values = np.nan_to_num(pred_df[feature].values.astype(float))
                max_val = float(np.max(feature_values)) if feature_values.size else 0.0
                thresholds = np.linspace(0.0, max_val if max_val > 0 else 1.0, 10)
                best_f1 = 0.0
                best_threshold = float(thresholds[0]) if thresholds.size else 0.0
                
                for threshold in thresholds:
                    binary_pred = (feature_values >= threshold).astype(int)
                    if len(set(binary_pred)) > 1:
                        precision, recall, f1, _ = precision_recall_fscore_support(
                            ground_truth, binary_pred, average='binary', zero_division=0
                        )
                        if f1 > best_f1:
                            best_f1 = float(f1)
                            best_threshold = float(threshold)
                
                results['threshold_analysis'][feature] = {
                    'best_threshold': best_threshold,
                    'best_f1': best_f1
                }
        
        # Feature importance with logistic regression
        try:
            if len(set(ground_truth)) > 1:
                lr = LogisticRegression(random_state=42, max_iter=1000)
                X = pred_df.fillna(0.0).to_numpy(dtype=float)
                y = np.array(ground_truth, dtype=int)
                lr.fit(X, y)
                feature_importance = dict(zip(pred_df.columns, lr.coef_[0]))
                results['feature_importance'] = {k: float(v) for k, v in feature_importance.items()}
                probs = lr.predict_proba(X)[:, 1]
                results['auc_score'] = float(roc_auc_score(y, probs))
        except Exception as e:
            results['logistic_regression_error'] = str(e)
        
        return results
    
    def run_ablation_study(self, split: str = "dev", model_type: str = "model-agnostic",
                          max_samples: int = 200) -> Dict[str, Any]:
        """
        Run ablation study comparing different feature combinations
        """
        # Map dev → val if needed
        actual_split = split 
        if split == "dev":
            try:
                _ = self.data_loader.extract_hypothesis_texts("dev", model_type)
            except FileNotFoundError:
                print("⚠️ Dev split not found, falling back to val split...")
                actual_split = "val"

        print(f"Running ablation study on {actual_split} split...")
        
        # Get evaluation results
        full_results = self.evaluate_on_split(actual_split, model_type, max_samples)
        
        if 'detailed_results' not in full_results:
            return {'error': 'No detailed results available for ablation'}
        
        detailed_results = full_results['detailed_results']
        
        # Feature combinations
        feature_combinations = {
            'spec_only': ['spec_score', 'hedge_count', 'absolute_count'],
            'sanity_only': ['num_sanity_score', 'total_flags'],
            'instability_only': ['instability_score'],
            'spec_sanity': ['spec_score', 'hedge_count', 'absolute_count', 'num_sanity_score', 'total_flags'],
            'all_features': ['spec_score', 'num_sanity_score', 'instability_score', 'total_flags', 
                           'hedge_count', 'absolute_count', 'entity_count', 'number_count', 'percent_count']
        }
        
        ablation_results = {}
        ground_truth = [1 if r['true_label'] == 'Hallucination' else 0 for r in detailed_results]
        
        for combo_name, features in feature_combinations.items():
            print(f"Testing feature combination: {combo_name}")
            try:
                feature_matrix = np.array([
                    [result['features'].get(feat, 0) for feat in features]
                    for result in detailed_results
                ], dtype=float)
                
                if len(set(ground_truth)) > 1 and feature_matrix.shape[1] > 0:
                    lr = LogisticRegression(random_state=42, max_iter=1000)
                    lr.fit(feature_matrix, ground_truth)
                    predictions = lr.predict(feature_matrix)
                    probabilities = lr.predict_proba(feature_matrix)[:, 1]
                    precision, recall, f1, _ = precision_recall_fscore_support(
                        ground_truth, predictions, average='binary', zero_division=0
                    )
                    auc = roc_auc_score(ground_truth, probabilities)
                    ablation_results[combo_name] = {
                        'precision': float(precision),
                        'recall': float(recall),
                        'f1': float(f1),
                        'auc': float(auc),
                        'features_used': features,
                        'feature_count': len(features)
                    }
                else:
                    ablation_results[combo_name] = {
                        'error': 'Not enough class variety for logistic regression'
                    }
            except Exception as e:
                ablation_results[combo_name] = {'error': str(e)}
        
        best_combo = None
        best_f1 = -1.0
        for name, metrics in ablation_results.items():
            f1 = metrics.get('f1', None)
            if isinstance(f1, (float, int)) and f1 > best_f1:
                best_f1 = f1
                best_combo = name
        
        return {'ablation_results': ablation_results, 'best_combination': best_combo}
    
    def generate_evaluation_report(self, results: Dict[str, Any], 
                                 output_file: Optional[str] = None) -> str:
        """Generate a comprehensive evaluation report"""
        report_lines = [
            "Claims & Sanity Pipeline - SHROOM Evaluation Report",
            "=" * 60,
            f"Dataset: {results.get('split', 'Unknown')} split ({results.get('model_type', 'Unknown')})",
            f"Total samples: {results.get('total_samples', 0)}",
            "",
            "Feature Correlations with Hallucination Label:",
            "-" * 45
        ]
        correlations = results.get('feature_correlations', {})
        for feature, corr in sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True):
            report_lines.append(f"{feature:20} {corr:+.4f}")
        
        report_lines.extend(["", "Threshold Analysis (Best F1 scores):", "-" * 37])
        threshold_analysis = results.get('threshold_analysis', {})
        for feature, analysis in threshold_analysis.items():
            threshold = analysis.get('best_threshold', 0)
            f1 = analysis.get('best_f1', 0)
            report_lines.append(f"{feature:20} threshold={threshold:.3f}, F1={f1:.3f}")
        
        if 'feature_importance' in results and results['feature_importance']:
            report_lines.extend(["", "Feature Importance (Logistic Regression):", "-" * 41])
            importance = results['feature_importance']
            for feature, coef in sorted(importance.items(), key=lambda x: abs(x[1]), reverse=True):
                report_lines.append(f"{feature:20} {coef:+.4f}")
        
        if 'auc_score' in results:
            report_lines.extend(["", f"Overall AUC Score: {results['auc_score']:.4f}"])
        
        report_text = "\n".join(report_lines)
        if output_file:
            with open(output_file, 'w') as f:
                f.write(report_text)
            print(f"Report saved to {output_file}")
        return report_text


def demo():
    """Demo the evaluator"""
    print("Pipeline Evaluator Demo")
    print("=" * 30)
    
    try:
        loader = SHROOMLoader()
        pipeline = ClaimsPipeline()
        evaluator = PipelineEvaluator(pipeline, loader)
        
        available = loader.get_available_splits()
        print(f"Available data: {available}")
        
        if available['splits']:
            results = evaluator.evaluate_on_split(
                split=available['splits'][0],
                model_type=available['model_types'][0] if available['model_types'] else 'model-agnostic',
                max_samples=50
            )
            report = evaluator.generate_evaluation_report(results)
            print("\nEvaluation Report:")
            print(report)
            
            print("\nRunning ablation study...")
            ablation = evaluator.run_ablation_study(max_samples=30)
            
            print("\nAblation Results:")
            for combo, metrics in ablation.get('ablation_results', {}).items():
                if 'error' not in metrics:
                    print(f"{combo:15} F1={metrics.get('f1', 0):.3f} AUC={metrics.get('auc', 0):.3f}")
                else:
                    print(f"{combo:15} ERROR: {metrics['error']}")
        
    except Exception as e:
        print(f"Demo error: {e}")


if __name__ == "__main__":
    demo()