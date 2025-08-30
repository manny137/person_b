#!/usr/bin/env python3
"""
benchmark.py - Full dataset benchmarking and Rule-as-data novelty lever
Implements auto-generation of silver labels and logistic regression training
"""

import json
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Tuple
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import brier_score_loss, precision_recall_fscore_support
from sklearn.calibration import calibration_curve
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import yaml

from data_loader import SHROOMLoader
from pipeline import ClaimsPipeline
from evaluator import PipelineEvaluator


class RuleAsDataNovelty:
    """
    Novelty Lever 1: Rule-as-data
    Auto-generate silver labels from rules.yaml, then train logistic regression
    on bag-of-ngrams for hedge detection with calibration (Brier score)
    """
    
    def __init__(self, pipeline: ClaimsPipeline, rules_path: str = "rules.yaml"):
        """Initialize with pipeline and rules"""
        self.pipeline = pipeline
        
        with open(rules_path, 'r') as f:
            config = yaml.safe_load(f)
        
        self.hedges = set(config['speculative']['hedges'])
        self.absolutes = set(config['speculative']['absolutes'])
        self.vectorizer = CountVectorizer(ngram_range=(1, 2), max_features=5000, stop_words='english')
        self.hedge_classifier = None
        self.silver_labels = []
        
    def generate_silver_labels(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Generate silver labels using rule-based pipeline
        
        Args:
            texts: List of input texts
            
        Returns:
            List of silver label dictionaries
        """
        print(f"Generating silver labels for {len(texts)} texts...")
        
        silver_labels = []
        
        for i, text in enumerate(texts):
            if i % 500 == 0:
                print(f"Processed {i}/{len(texts)} texts...")
            
            try:
                # Run pipeline to get rule-based features
                result = self.pipeline.process_text(text)
                
                if result.get('sentences'):
                    sent_result = result['sentences'][0]
                    
                    # Extract silver label features
                    silver_label = {
                        'text': text,
                        'has_hedges': sent_result['spec_counts']['hedges'] > 0,
                        'has_absolutes': sent_result['spec_counts']['absolutes'] > 0,
                        'hedge_count': sent_result['spec_counts']['hedges'],
                        'absolute_count': sent_result['spec_counts']['absolutes'],
                        'spec_score': sent_result['spec_score'],
                        'num_sanity_score': sent_result['num_sanity_score'],
                        'instability_score': sent_result['instability_score'],
                        'total_flags': len(sent_result['flags']),
                        
                        # Silver label: high speculation if spec_score > 0.1 or has hedges
                        'silver_speculative': (sent_result['spec_score'] > 0.1 or 
                                             sent_result['spec_counts']['hedges'] > 0),
                        
                        # Silver label: potentially hallucinated if high sanity issues or speculation
                        'silver_hallucination': (sent_result['num_sanity_score'] > 0.3 or
                                                sent_result['spec_score'] > 0.2 or
                                                len(sent_result['flags']) > 1)
                    }
                    
                    silver_labels.append(silver_label)
                    
            except Exception as e:
                # Fallback for failed processing
                silver_labels.append({
                    'text': text,
                    'has_hedges': False,
                    'has_absolutes': False,
                    'hedge_count': 0,
                    'absolute_count': 0,
                    'spec_score': 0.0,
                    'num_sanity_score': 0.0,
                    'instability_score': 0.0,
                    'total_flags': 0,
                    'silver_speculative': False,
                    'silver_hallucination': False
                })
        
        self.silver_labels = silver_labels
        print(f"Generated {len(silver_labels)} silver labels")
        return silver_labels
    
    def train_ngram_classifier(self, texts: List[str], silver_labels: List[Dict]) -> Dict[str, Any]:
        """
        Train logistic regression on bag-of-ngrams using silver labels
        
        Args:
            texts: Input texts
            silver_labels: Generated silver labels
            
        Returns:
            Training results and metrics
        """
        print("Training n-gram classifier on silver labels...")
        
        # Prepare features and labels
        X = self.vectorizer.fit_transform(texts)
        y_speculative = [label['silver_speculative'] for label in silver_labels]
        y_hallucination = [label['silver_hallucination'] for label in silver_labels]
        
        results = {}
        
        # Train hedge/speculation classifier
        if len(set(y_speculative)) > 1:  # Need both classes
            self.hedge_classifier = LogisticRegression(random_state=42, max_iter=1000)
            self.hedge_classifier.fit(X, y_speculative)
            
            # Cross-validation scores
            cv_scores = cross_val_score(self.hedge_classifier, X, y_speculative, cv=5, scoring='f1')
            
            # Calibration analysis
            probs = self.hedge_classifier.predict_proba(X)[:, 1]
            brier_score = brier_score_loss(y_speculative, probs)
            
            results['hedge_classifier'] = {
                'cv_f1_mean': np.mean(cv_scores),
                'cv_f1_std': np.std(cv_scores),
                'brier_score': brier_score,
                'feature_count': X.shape[1],
                'positive_rate': np.mean(y_speculative),
                'calibration_analysis': self._analyze_calibration(y_speculative, probs)
            }
        
        # Train hallucination classifier
        if len(set(y_hallucination)) > 1:
            halluc_classifier = LogisticRegression(random_state=42, max_iter=1000)
            halluc_classifier.fit(X, y_hallucination)
            
            cv_scores_h = cross_val_score(halluc_classifier, X, y_hallucination, cv=5, scoring='f1')
            probs_h = halluc_classifier.predict_proba(X)[:, 1]
            brier_score_h = brier_score_loss(y_hallucination, probs_h)
            
            results['hallucination_classifier'] = {
                'cv_f1_mean': np.mean(cv_scores_h),
                'cv_f1_std': np.std(cv_scores_h),
                'brier_score': brier_score_h,
                'positive_rate': np.mean(y_hallucination),
                'calibration_analysis': self._analyze_calibration(y_hallucination, probs_h)
            }
        
        return results
    
    def _analyze_calibration(self, y_true: List[int], y_prob: np.ndarray) -> Dict[str, float]:
        """Analyze calibration of probability predictions"""
        try:
            fraction_of_positives, mean_predicted_value = calibration_curve(
                y_true, y_prob, n_bins=5
            )
            
            # Calculate calibration error (Expected Calibration Error)
            bin_boundaries = np.linspace(0, 1, 6)
            bin_lowers = bin_boundaries[:-1]
            bin_uppers = bin_boundaries[1:]
            
            ece = 0
            for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
                in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
                prop_in_bin = in_bin.mean()
                
                if prop_in_bin > 0:
                    accuracy_in_bin = np.array(y_true)[in_bin].mean()
                    avg_confidence_in_bin = y_prob[in_bin].mean()
                    ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
            
            return {
                'expected_calibration_error': ece,
                'reliability_bins': len(fraction_of_positives),
                'mean_confidence': np.mean(y_prob),
                'mean_accuracy': np.mean(y_true)
            }
            
        except Exception:
            return {'error': 'Could not calculate calibration metrics'}
    
    def predict_with_ngrams(self, texts: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict using trained n-gram classifier
        
        Args:
            texts: Input texts to predict on
            
        Returns:
            Tuple of (predictions, probabilities)
        """
        if self.hedge_classifier is None:
            raise ValueError("Classifier not trained yet. Call train_ngram_classifier first.")
        
        X = self.vectorizer.transform(texts)
        predictions = self.hedge_classifier.predict(X)
        probabilities = self.hedge_classifier.predict_proba(X)[:, 1]
        
        return predictions, probabilities
    
    def compare_rule_vs_ngram(self, test_texts: List[str], true_labels: List[int]) -> Dict[str, Any]:
        """
        Compare rule-based approach vs n-gram classifier
        
        Args:
            test_texts: Test texts
            true_labels: Ground truth labels (1 for hallucination, 0 for not)
            
        Returns:
            Comparison results
        """
        print("Comparing rule-based vs n-gram approaches...")
        
        # Get rule-based predictions
        rule_predictions = []
        for text in test_texts:
            result = self.pipeline.process_text(text)
            if result.get('sentences'):
                # Use combination of spec_score and sanity_score as prediction
                sent = result['sentences'][0]
                rule_score = (sent['spec_score'] + sent['num_sanity_score'] + 
                             sent['instability_score']) / 3
                rule_predictions.append(1 if rule_score > 0.2 else 0)
            else:
                rule_predictions.append(0)
        
        # Get n-gram predictions
        ngram_preds, ngram_probs = self.predict_with_ngrams(test_texts)
        
        # Calculate metrics for both
        rule_precision, rule_recall, rule_f1, _ = precision_recall_fscore_support(
            true_labels, rule_predictions, average='binary', zero_division=0
        )
        
        ngram_precision, ngram_recall, ngram_f1, _ = precision_recall_fscore_support(
            true_labels, ngram_preds, average='binary', zero_division=0
        )
        
        comparison = {
            'rule_based': {
                'precision': rule_precision,
                'recall': rule_recall,
                'f1': rule_f1
            },
            'ngram_classifier': {
                'precision': ngram_precision,
                'recall': ngram_recall,
                'f1': ngram_f1,
                'brier_score': brier_score_loss(true_labels, ngram_probs)
            },
            'improvement': {
                'f1_delta': ngram_f1 - rule_f1,
                'precision_delta': ngram_precision - rule_precision,
                'recall_delta': ngram_recall - rule_recall
            }
        }
        
        return comparison


class BenchmarkRunner:
    """Run comprehensive benchmarks on SHROOM dataset"""
    
    def __init__(self):
        """Initialize benchmark runner"""
        self.loader = SHROOMLoader()
        self.pipeline = ClaimsPipeline()
        self.evaluator = PipelineEvaluator(self.pipeline, self.loader)
        self.rule_as_data = RuleAsDataNovelty(self.pipeline)
    
    def run_full_benchmark(self, max_samples: int = 1000) -> Dict[str, Any]:
        """
        Run complete benchmark including rule-as-data novelty
        
        Args:
            max_samples: Maximum samples per split for testing
            
        Returns:
            Complete benchmark results
        """
        print("Starting full benchmark...")
        
        results = {
            'benchmark_config': {
                'max_samples_per_split': max_samples,
                'timestamp': pd.Timestamp.now().isoformat()
            }
        }
        
        available = self.loader.get_available_splits()
        print(f"Available datasets: {available}")
        
        # Test each available split
        for split in available.get('splits', []):
            for model_type in available.get('model_types', ['model-agnostic']):
                print(f"\n--- Benchmarking {split} split ({model_type}) ---")
                
                try:
                    # Standard evaluation
                    eval_results = self.evaluator.evaluate_on_split(
                        split=split, model_type=model_type, max_samples=max_samples
                    )
                    
                    # Ablation study
                    ablation_results = self.evaluator.run_ablation_study(
                        split=split, model_type=model_type, max_samples=max_samples//2
                    )
                    
                    # Rule-as-data novelty
                    texts_labels = self.loader.extract_hypothesis_texts(split, model_type)
                    if max_samples:
                        texts_labels = texts_labels[:max_samples]
                    
                    texts = [t for t, l in texts_labels]
                    labels = [1 if l == 'Hallucination' else 0 for t, l in texts_labels]
                    
                    # Generate silver labels and train classifier
                    silver_labels = self.rule_as_data.generate_silver_labels(texts)
                    training_results = self.rule_as_data.train_ngram_classifier(texts, silver_labels)
                    
                    # Compare approaches
                    comparison = self.rule_as_data.compare_rule_vs_ngram(texts, labels)
                    
                    # Store results
                    key = f"{split}_{model_type}"
                    results[key] = {
                        'evaluation': eval_results,
                        'ablation': ablation_results,
                        'rule_as_data': {
                            'training_results': training_results,
                            'comparison': comparison,
                            'silver_labels_generated': len(silver_labels)
                        }
                    }
                    
                except Exception as e:
                    print(f"Error benchmarking {split} {model_type}: {e}")
                    results[f"{split}_{model_type}"] = {'error': str(e)}
        
        return results
    
    def generate_benchmark_report(self, results: Dict[str, Any], 
                                output_file: str = "benchmark_report.txt") -> str:
        """Generate comprehensive benchmark report"""
        
        report_lines = [
            "CLAIMS & SANITY PIPELINE - COMPREHENSIVE BENCHMARK REPORT",
            "=" * 70,
            f"Generated: {results.get('benchmark_config', {}).get('timestamp', 'Unknown')}",
            f"Max samples per split: {results.get('benchmark_config', {}).get('max_samples_per_split', 'All')}",
            ""
        ]
        
        # Summary statistics
        report_lines.extend([
            "SUMMARY STATISTICS",
            "-" * 20,
        ])
        
        splits_processed = 0
        best_f1 = 0
        best_config = ""
        
        for key, result in results.items():
            if key.startswith('benchmark_config'):
                continue
                
            if 'error' in result:
                report_lines.append(f"{key}: ERROR - {result['error']}")
                continue
            
            splits_processed += 1
            
            # Get best F1 from ablation study
            if 'ablation' in result and 'ablation_results' in result['ablation']:
                ablation = result['ablation']['ablation_results']
                for combo, metrics in ablation.items():
                    if 'f1' in metrics and metrics['f1'] > best_f1:
                        best_f1 = metrics['f1']
                        best_config = f"{key}_{combo}"
        
        report_lines.extend([
            f"Splits processed: {splits_processed}",
            f"Best F1 score: {best_f1:.4f} ({best_config})",
            ""
        ])
        
        # Detailed results for each split
        for key, result in results.items():
            if key.startswith('benchmark_config') or 'error' in result:
                continue
            
            report_lines.extend([
                f"RESULTS FOR {key.upper()}",
                "-" * (12 + len(key)),
            ])
            
            # Feature correlations
            if 'evaluation' in result and 'feature_correlations' in result['evaluation']:
                report_lines.append("Feature correlations with hallucination:")
                correlations = result['evaluation']['feature_correlations']
                for feat, corr in sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)[:5]:
                    report_lines.append(f"  {feat:20} {corr:+.4f}")
            
            # Ablation study results
            if 'ablation' in result and 'ablation_results' in result['ablation']:
                report_lines.append("\nAblation study results:")
                ablation = result['ablation']['ablation_results']
                for combo, metrics in ablation.items():
                    if 'f1' in metrics:
                        report_lines.append(f"  {combo:15} F1={metrics['f1']:.3f} AUC={metrics.get('auc', 0):.3f}")
            
            # Rule-as-data novelty results
            if 'rule_as_data' in result:
                rad = result['rule_as_data']
                report_lines.append("\nRule-as-data novelty results:")
                
                if 'training_results' in rad:
                    training = rad['training_results']
                    if 'hedge_classifier' in training:
                        hc = training['hedge_classifier']
                        report_lines.append(f"  Hedge classifier: F1={hc.get('cv_f1_mean', 0):.3f} ± {hc.get('cv_f1_std', 0):.3f}")
                        report_lines.append(f"  Brier score: {hc.get('brier_score', 0):.4f}")
                        
                        if 'calibration_analysis' in hc:
                            cal = hc['calibration_analysis']
                            if 'expected_calibration_error' in cal:
                                report_lines.append(f"  Expected Calibration Error: {cal['expected_calibration_error']:.4f}")
                
                if 'comparison' in rad:
                    comp = rad['comparison']
                    report_lines.append("  Rule-based vs N-gram comparison:")
                    if 'rule_based' in comp:
                        rb = comp['rule_based']
                        report_lines.append(f"    Rules only:  F1={rb.get('f1', 0):.3f}")
                    if 'ngram_classifier' in comp:
                        ng = comp['ngram_classifier']
                        report_lines.append(f"    N-gram+Rules: F1={ng.get('f1', 0):.3f}")
                    if 'improvement' in comp:
                        imp = comp['improvement']
                        report_lines.append(f"    Improvement:  ΔF1={imp.get('f1_delta', 0):+.3f}")
            
            report_lines.append("")
        
        # Conclusions and recommendations
        report_lines.extend([
            "CONCLUSIONS & RECOMMENDATIONS",
            "-" * 32,
            f"• Best performing configuration: {best_config} (F1={best_f1:.4f})",
            "• Rule-as-data approach adds learned patterns to complement wordlists",
            "• Calibration analysis helps assess prediction confidence reliability",
            "• Instability scores from paraphrases provide additional signal",
            "",
            "NEXT STEPS",
            "-" * 10,
            "• Implement temporal anchoring for better date conflict detection",
            "• Add unit canonizer to reduce false absurdity flags",
            "• Create paraphrase stability visualization cards",
            "• Fine-tune thresholds based on calibration analysis"
        ])
        
        report_text = "\n".join(report_lines)
        
        # Save report
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        print(f"Benchmark report saved to {output_file}")
        return report_text


def main():
    """CLI for running benchmarks"""
    parser = argparse.ArgumentParser(description="Run SHROOM benchmark")
    parser.add_argument("--max-samples", "-m", type=int, default=1000,
                       help="Maximum samples per split for testing")
    parser.add_argument("--output", "-o", default="benchmark_results.json",
                       help="Output file for results")
    parser.add_argument("--report", "-r", default="benchmark_report.txt",
                       help="Output file for human-readable report")
    parser.add_argument("--quick", "-q", action="store_true",
                       help="Quick test with 100 samples")
    
    args = parser.parse_args()
    
    if args.quick:
        args.max_samples = 100
        print("Quick test mode: using 100 samples per split")
    
    # Run benchmark
    runner = BenchmarkRunner()
    results = runner.run_full_benchmark(max_samples=args.max_samples)
    
    # Save results
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"Benchmark results saved to {args.output}")
    
    # Generate report
    report = runner.generate_benchmark_report(results, args.report)
    
    print("\nBenchmark completed!")
    print(f"Results: {args.output}")
    print(f"Report: {args.report}")


if __name__ == "__main__":
    main()