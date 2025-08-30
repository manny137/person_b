#!/usr/bin/env python3
"""
pipeline.py - Main CLI orchestrator
Wires together all components: parser, speculative, sanity, paraphrase
"""

import sys
import json
import argparse
from typing import Dict, Any, List
from parser import ClaimExtractor
from speculative import SpeculativeScorer
from sanity import SanityChecker
from paraphrase import ParaphraseGenerator


class ClaimsPipeline:
    """Main pipeline combining all analysis components"""
    
    def __init__(self, rules_path: str = "rules.yaml"):
        """Initialize all components"""
        self.extractor = ClaimExtractor()
        self.spec_scorer = SpeculativeScorer(rules_path)
        self.sanity_checker = SanityChecker(rules_path)
        self.paraphraser = ParaphraseGenerator(rules_path)
    
    def process_text(self, text: str) -> Dict[str, Any]:
        """
        Process text through the full pipeline
        
        Args:
            text: Input text to analyze
            
        Returns:
            Complete analysis results in JSON format
        """
        # Step 1: Extract claims from sentences

        
        sentences_data = self.extractor.extract_sentence_claims(text)
        
        results = {"sentences": [], "summary": {}}
        spec_scores = []
        sanity_scores = []
        instability_scores = []
        
        # Step 2: Process each sentence
        for sent_data in sentences_data:
            sent_text = sent_data['text']
            
            # B2: Calculate speculative score
            spec_score, spec_counts = self.spec_scorer.score_sentence(sent_text)
            
            # B3: Check numeric sanity
            num_sanity_score, flags = self.sanity_checker.check_sentence_claims(sent_data)
            
            # B4: Generate paraphrases and calculate instability
            paraphrases = self.paraphraser.generate_paraphrases(sent_text)
            
            # Score paraphrases
            paraphrase_scores = []
            for para in paraphrases:
                para_spec, _ = self.spec_scorer.score_sentence(para)
                # Create dummy sentence data for sanity checking paraphrases
                para_sent_data = self.extractor.extract_sentence_claims(para)[0] if self.extractor.extract_sentence_claims(para) else sent_data
                para_sanity, _ = self.sanity_checker.check_sentence_claims(para_sent_data)
                
                paraphrase_scores.append({
                    'spec_score': para_spec,
                    'num_sanity_score': para_sanity
                })
            
            original_scores = {
                'spec_score': spec_score,
                'num_sanity_score': num_sanity_score
            }
            
            instability_score = self.paraphraser.calculate_instability(
                original_scores, paraphrase_scores
            )
            
            # Compile sentence results
            sentence_result = {
                "text": sent_text,
                "claims": sent_data['claims'],
                "spec_score": round(spec_score, 3),
                "num_sanity_score": round(num_sanity_score, 3),
                "instability_score": round(instability_score, 3),
                "flags": flags,
                "paraphrases": paraphrases,
                "spec_counts": spec_counts
            }
            
            results["sentences"].append(sentence_result)
            
            # Collect for summary
            spec_scores.append(spec_score)
            sanity_scores.append(num_sanity_score)
            instability_scores.append(instability_score)
        
        # Step 3: Calculate summary statistics
        results["summary"] = {
            "avg_spec": round(sum(spec_scores) / len(spec_scores) if spec_scores else 0, 3),
            "avg_num_sanity": round(sum(sanity_scores) / len(sanity_scores) if sanity_scores else 0, 3),
            "avg_instability": round(sum(instability_scores) / len(instability_scores) if instability_scores else 0, 3),
            "total_sentences": len(sentences_data),
            "total_flags": sum(len(s["flags"]) for s in results["sentences"])
        }
        
        return results


def main():
    """CLI entry point"""
    parser = argparse.ArgumentParser(description="Claims & Sanity Analysis Pipeline")
    parser.add_argument("--input", "-i", help="Input file path (default: stdin)")
    parser.add_argument("--output", "-o", help="Output file path (default: stdout)")
    parser.add_argument("--rules", "-r", default="rules.yaml", help="Rules configuration file")
    parser.add_argument("--pretty", "-p", action="store_true", help="Pretty print JSON output")
    
    args = parser.parse_args()
    
    # Initialize pipeline
    try:
        pipeline = ClaimsPipeline(args.rules)
    except Exception as e:
        print(f"Error initializing pipeline: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Read input
    if args.input:
        try:
            with open(args.input, 'r', encoding='utf-8') as f:
                text = f.read().strip()
        except Exception as e:
            print(f"Error reading input file: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        text = sys.stdin.read().strip()
    
    if not text:
        print("Error: No input text provided", file=sys.stderr)
        sys.exit(1)
    
    # Process text
    try:
        results = pipeline.process_text(text)
    except Exception as e:
        print(f"Error processing text: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Write output
    json_output = json.dumps(results, indent=2 if args.pretty else None, ensure_ascii=False)
    
    if args.output:
        try:
            with open(args.output, 'w', encoding='utf-8') as f:
                f.write(json_output)
        except Exception as e:
            print(f"Error writing output file: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        print(json_output)


def demo():
    """Demo the complete pipeline"""
    pipeline = ClaimsPipeline()
    
    test_text = """
    Apple Inc. (AAPL) definitely reported exceptional earnings that increased 25% in Q2 2024.
    The stock price jumped an unbelievable 500% in one day, reaching $250 billion market cap.
    Analysts suggest the company might see continued growth, possibly driven by new products.
    """
    
    print("Complete Pipeline Demo:")
    print("=" * 60)
    print(f"Input text:\n{test_text}")
    print("\n" + "=" * 60)
    
    results = pipeline.process_text(test_text)
    
    print("Results:")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    if len(sys.argv) == 1:
        # No arguments provided, run demo
        demo()
    else:
        main()