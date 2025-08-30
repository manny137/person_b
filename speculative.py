#!/usr/bin/env python3
"""
speculative.py - B2: Speculative language score
Measures hedges and absolutes in text using configurable wordlists
"""

import yaml
import re
from typing import Dict, List, Tuple
import spacy


class SpeculativeScorer:
    """Score speculative language using hedge and absolute word detection"""
    
    def __init__(self, rules_path: str = "rules.yaml"):
        """Initialize with rules from YAML config"""
        with open(rules_path, 'r') as f:
            config = yaml.safe_load(f)
        
        self.spec_config = config['speculative']
        self.hedges = set(self.spec_config['hedges'])
        self.absolutes = set(self.spec_config['absolutes'])
        self.weights = self.spec_config['weights']
        
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            raise RuntimeError("Please install spaCy English model")
    
    def score_sentence(self, text: str) -> Tuple[float, Dict[str, int]]:
        """
        Calculate speculative score for a sentence
        
        Args:
            text: Input sentence
            
        Returns:
            Tuple of (spec_score, word_counts)
        """
        # Tokenize and normalize
        doc = self.nlp(text.lower())
        tokens = [token.lemma_ for token in doc if not token.is_punct and not token.is_space]
        
        if not tokens:
            return 0.0, {'hedges': 0, 'absolutes': 0, 'tokens': 0}
        
        # Count hedges and absolutes
        hedge_count = sum(1 for token in tokens if token in self.hedges)
        absolute_count = sum(1 for token in tokens if token in self.absolutes)
        
        # Calculate weighted score
        weighted_sum = (hedge_count * self.weights['hedge'] + 
                       absolute_count * self.weights['absolute'])
        
        # Normalize by token count with dampening factor
        spec_score = min(weighted_sum / (0.02 * len(tokens)), 1.0)
        
        counts = {
            'hedges': hedge_count,
            'absolutes': absolute_count, 
            'tokens': len(tokens)
        }
        
        return spec_score, counts
    
    def get_matched_words(self, text: str) -> Dict[str, List[str]]:
        """Return the actual hedge/absolute words found in text"""
        doc = self.nlp(text.lower())
        tokens = [token.lemma_ for token in doc if not token.is_punct and not token.is_space]
        
        matched_hedges = [token for token in tokens if token in self.hedges]
        matched_absolutes = [token for token in tokens if token in self.absolutes]
        
        return {
            'hedges': matched_hedges,
            'absolutes': matched_absolutes
        }


def demo():
    """Demo the speculative scorer"""
    scorer = SpeculativeScorer()
    
    test_sentences = [
        "Apple definitely increased revenue by 15% last quarter.",  # absolute
        "The company might see growth, possibly reaching new highs.",  # hedges
        "Revenue grew 15% in Q2 according to the earnings report.",  # neutral
        "This investment is guaranteed to always provide returns without doubt."  # high absolute
    ]
    
    print("Speculative Language Analysis:")
    print("-" * 50)
    
    for sent in test_sentences:
        score, counts = scorer.score_sentence(sent)
        matches = scorer.get_matched_words(sent)
        
        print(f"\nSentence: {sent}")
        print(f"Score: {score:.3f}")
        print(f"Counts: {counts}")
        print(f"Hedges found: {matches['hedges']}")
        print(f"Absolutes found: {matches['absolutes']}")


if __name__ == "__main__":
    demo()