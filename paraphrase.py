#!/usr/bin/env python3
"""
paraphrase.py - B4: Self-consistency via paraphrases
Creates deterministic paraphrases and measures stability
"""

import yaml
import re
import random
from typing import List, Dict, Any, Tuple
import spacy
import numpy as np


class ParaphraseGenerator:
    """Generate deterministic paraphrases for stability testing"""
    
    def __init__(self, rules_path: str = "rules.yaml"):
        """Initialize with paraphrase rules from config"""
        with open(rules_path, 'r') as f:
            config = yaml.safe_load(f)
        
        self.synonyms = config['paraphrase']['synonyms']
        
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            raise RuntimeError("Please install spaCy English model")
        
        # Fixed seed for deterministic paraphrases
        random.seed(42)
    
    def generate_paraphrases(self, text: str) -> List[str]:
        """
        Generate 3 deterministic paraphrases using different strategies
        
        Args:
            text: Original sentence
            
        Returns:
            List of 3 paraphrases
        """
        paraphrases = []
        
        # Paraphrase 1: Synonym substitution
        paraphrases.append(self._synonym_paraphrase(text))
        
        # Paraphrase 2: Active/Passive voice swap  
        paraphrases.append(self._voice_paraphrase(text))
        
        # Paraphrase 3: Clause reordering
        paraphrases.append(self._clause_reorder_paraphrase(text))
        
        return paraphrases
    
    def _synonym_paraphrase(self, text: str) -> str:
        """Replace words with synonyms from the dictionary"""
        result = text.lower()
        
        # Apply synonym substitutions deterministically
        for word, synonyms in self.synonyms.items():
            if word in result:
                # Always use first synonym for deterministic output
                result = result.replace(word, synonyms[0])
        
        # Capitalize first letter
        return result[0].upper() + result[1:] if result else result

    def _voice_paraphrase(self, text: str) -> str:
        """Convert between active and passive voice (simplified)"""
        doc = self.nlp(text)
        result = text
        
        # Detect passive
        has_passive = any(token.dep_ == "auxpass" for token in doc)
        
        if not has_passive:
            # Attempt simple active → passive
            for sent in doc.sents:
                subj = None
                verb = None
                obj = None
                
                for token in sent:
                    if token.dep_ == "nsubj":
                        subj = token
                    elif token.pos_ == "VERB" and token.dep_ == "ROOT":
                        verb = token
                    elif token.dep_ == "dobj":
                        obj = token
                
                if subj and verb and obj:
                    passive_form = f"{obj.text} was {verb.lemma_}ed by {subj.text}"
                    result = result.replace(sent.text.strip(), passive_form)
        # If already passive, return as is for now
        return result
                    
    def _clause_reorder_paraphrase(self, text: str) -> str:
        """Reorder clauses using conjunctions and relative pronouns"""
        conjunctions = [', which', ', that', ' and ', ' but ', ' while ']
        
        for conj in conjunctions:
            if conj in text:
                parts = text.split(conj, 1)
                if len(parts) == 2:
                    # Reorder: move second clause to beginning
                    if conj.startswith(','):
                        return f"{parts[1]}, and {parts[0].lower()}"
                    else:
                        return f"{parts[1].strip()}, {parts[0].lower()}"
        
        # If no conjunctions, try moving prepositional phrases
        doc = self.nlp(text)
        prep_phrases = []
        main_clause = text
        
        for chunk in doc.noun_chunks:
            if any(token.dep_ == "prep" for token in chunk):
                prep_phrases.append(chunk.text)
        
        if prep_phrases:
            # Move first prepositional phrase to end
            phrase = prep_phrases[0]
            main_clause = main_clause.replace(phrase, "").strip()
            return f"{main_clause} {phrase}"
        
        return text
    
    def calculate_instability(self, original_scores: Dict[str, float], 
                            paraphrase_scores: List[Dict[str, float]]) -> float:
        """
        Calculate instability score based on variance across paraphrases
        
        Args:
            original_scores: Scores from original sentence
            paraphrase_scores: List of scores from each paraphrase
            
        Returns:
            Instability score (0-1)
        """
        all_scores = [original_scores] + paraphrase_scores
        
        # Calculate variance for each score type
        score_types = ['spec_score', 'num_sanity_score']
        variances = []
        
        for score_type in score_types:
            if score_type in original_scores:
                values = [scores.get(score_type, 0) for scores in all_scores]
                if len(set(values)) > 1:  # Only if there's variation
                    variance = np.var(values)
                    variances.append(variance)
        
        if not variances:
            return 0.0
        
        # Average variance across score types
        avg_variance = np.mean(variances)
        
        # Normalize using the formula: min(Var(risk_base)/0.06, 1.0)
        instability_score = min(avg_variance / 0.06, 1.0)
        
        return instability_score


def demo():
    """Demo the paraphrase generator and instability calculation"""
    generator = ParaphraseGenerator()
    
    test_sentence = "Apple Inc. reported strong earnings, which increased 25% in the last quarter."
    
    print("Paraphrase Generation Demo:")
    print("-" * 50)
    print(f"Original: {test_sentence}")
    
    paraphrases = generator.generate_paraphrases(test_sentence)
    
    for i, paraphrase in enumerate(paraphrases, 1):
        print(f"Paraphrase {i}: {paraphrase}")
    
    print("\nInstability Calculation Demo:")
    print("-" * 30)
    
    original_scores = {'spec_score': 0.2, 'num_sanity_score': 0.1}
    paraphrase_scores = [
        {'spec_score': 0.25, 'num_sanity_score': 0.1},
        {'spec_score': 0.18, 'num_sanity_score': 0.15}, 
        {'spec_score': 0.22, 'num_sanity_score': 0.08}
    ]
    
    instability = generator.calculate_instability(original_scores, paraphrase_scores)
    
    print(f"Original scores: {original_scores}")
    print(f"Paraphrase scores: {paraphrase_scores}")
    print(f"Instability score: {instability:.3f}")


if __name__ == "__main__":
    demo()