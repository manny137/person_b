#!/usr/bin/env python3
"""
parser.py - B1: Claim/slot extractor
Extracts entities, dates, numbers, tickers, percents using regex + spaCy
"""

import re
import spacy
from typing import List, Dict, Any, Tuple
import dateparser
from datetime import datetime


class ClaimExtractor:
    """Extract claims from text using spaCy + regex patterns"""
    
    def __init__(self):
        """Initialize with spaCy model and regex patterns"""
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            raise RuntimeError("Please install spaCy English model: python -m spacy download en_core_web_sm")
        
        # Regex patterns
        self.patterns = {
            'ticker': re.compile(r'\b[A-Z]{1,5}(?:\.[A-Z]{1,3})?\b'),
            'percent': re.compile(r'-?\d+(?:\.\d+)?%'),
            'money': re.compile(r'[\$₹€£]\s*\d+(?:,\d{3})*(?:\.\d{2})?(?:\s*(?:billion|million|thousand|B|M|K))?'),
            'number': re.compile(r'-?\d+(?:,\d{3})*(?:\.\d+)?(?:\s*(?:billion|million|thousand|B|M|K))?'),
            'date': re.compile(r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4}|\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\b\d{4}[/-]\d{1,2}[/-]\d{1,2}|\bQ[1-4]\s+(?:FY)?\d{2,4}')
        }
    
    def extract_sentence_claims(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract claims from each sentence in text
        
        Args:
            text: Input text
            
        Returns:
            List of sentence claims with entities, numbers, dates, etc.
        """
        doc = self.nlp(text)
        sentences = []
        
        for sent in doc.sents:
            sent_text = sent.text.strip()
            if not sent_text:
                continue
                
            # Extract entities using spaCy
            entities = []
            for ent in sent.ents:
                if ent.label_ in ['ORG', 'PERSON', 'GPE', 'PRODUCT', 'MONEY']:
                    entities.append({
                        'text': ent.text,
                        'label': ent.label_,
                        'start': ent.start_char,
                        'end': ent.end_char
                    })
            
            # Extract patterns using regex
            claims = {
                'entities': entities,
                'tickers': self._extract_pattern('ticker', sent_text),
                'percents': self._extract_pattern('percent', sent_text),
                'money': self._extract_pattern('money', sent_text),
                'numbers': self._extract_pattern('number', sent_text),
                'dates': self._extract_dates(sent_text)
            }
            
            sentences.append({
                'text': sent_text,
                'start': sent.start_char,
                'end': sent.end_char,
                'claims': claims
            })
        
        return sentences
    
    def _extract_pattern(self, pattern_name: str, text: str) -> List[Dict[str, Any]]:
        """Extract matches for a specific pattern"""
        matches = []
        for match in self.patterns[pattern_name].finditer(text):
            matches.append({
                'text': match.group(),
                'start': match.start(),
                'end': match.end()
            })
        return matches
    
    def _extract_dates(self, text: str) -> List[Dict[str, Any]]:
        """Extract and parse dates from text"""
        dates = []
        for match in self.patterns['date'].finditer(text):
            date_text = match.group()
            parsed = dateparser.parse(date_text)
            dates.append({
                'text': date_text,
                'start': match.start(),
                'end': match.end(),
                'parsed': parsed.isoformat() if parsed else None
            })
        return dates


def demo():
    """Demo the claim extractor"""
    extractor = ClaimExtractor()
    
    test_text = """
    Apple Inc. (AAPL) reported 15.8% revenue growth in Q2 2024, reaching $123.5 billion.
    The company's stock jumped 25% on March 15, 2024 after the announcement.
    CEO Tim Cook said the results were "absolutely outstanding" with no doubt about future growth.
    """
    
    sentences = extractor.extract_sentence_claims(test_text)
    
    for i, sent in enumerate(sentences):
        print(f"\nSentence {i+1}: {sent['text']}")
        print(f"Entities: {len(sent['claims']['entities'])}")
        print(f"Tickers: {[t['text'] for t in sent['claims']['tickers']]}")
        print(f"Percents: {[p['text'] for p in sent['claims']['percents']]}")
        print(f"Money: {[m['text'] for m in sent['claims']['money']]}")
        print(f"Dates: {[d['text'] for d in sent['claims']['dates']]}")


if __name__ == "__main__":
    demo()