#!/usr/bin/env python3
"""
sanity.py - B3: Numeric & temporal sanity checks
Flags unrealistic numbers, dates, currencies, and units
"""

import yaml
import re
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime, timedelta
import dateparser


class SanityChecker:
    """Check numeric and temporal sanity of extracted claims"""
    
    def __init__(self, rules_path: str = "rules.yaml"):
        """Initialize with sanity rules from YAML config"""
        with open(rules_path, 'r') as f:
            config = yaml.safe_load(f)
        
        self.sanity_config = config['sanity']
        self.rules = self.sanity_config['rules']
        self.thresholds = self.sanity_config['thresholds']
        self.currencies = self.sanity_config['currencies']
    
    def check_sentence_claims(self, sentence_data: Dict[str, Any]) -> Tuple[float, List[str]]:
        """
        Check sanity of all claims in a sentence
        
        Args:
            sentence_data: Sentence data with extracted claims
            
        Returns:
            Tuple of (num_sanity_score, list_of_flags)
        """
        flags = []
        claims = sentence_data['claims']
        text = sentence_data['text']
        
        # Count total numeric claims
        numeric_claims = (len(claims['percents']) + len(claims['money']) + 
                         len(claims['numbers']) + len(claims['dates']))
        
        if numeric_claims == 0:
            return 0.0, []
        
        # Check each type of claim
        flags.extend(self._check_percent_jumps(claims['percents'], claims['dates']))
        flags.extend(self._check_currency_mismatch(claims['money'], text))
        flags.extend(self._check_unit_absurdity(claims['numbers'], text))
        flags.extend(self._check_temporal_conflicts(claims['dates'], text))
        
        # Calculate sanity score
        num_sanity_score = len(flags) / max(1, numeric_claims)
        
        return min(num_sanity_score, 1.0), flags
    
    def _check_percent_jumps(self, percents: List[Dict], dates: List[Dict]) -> List[str]:
        """Check for unrealistic percentage jumps in short time periods"""
        flags = []
        
        if not self.rules['percent_jump']['enabled'] or len(percents) < 1:
            return flags
        
        for percent in percents:
            percent_text = percent['text'].replace('%', '')
            try:
                value = abs(float(percent_text))
                if value > self.thresholds['percent_max'] and any('day' in d['text'].lower() for d in dates):
                    flags.append(f"percent_jump_{value}")
                elif value > 300 and any('day' in d['text'].lower() or 'today' in d['text'].lower() for d in dates):
                    flags.append(f"extreme_percent_jump_{value}")
            except ValueError:
                continue
        
        return flags
    
    def _check_currency_mismatch(self, money_claims: List[Dict], text: str) -> List[str]:
        """Check for currency symbol/context mismatches"""
        flags = []
        
        if not self.rules['currency_mismatch']['enabled']:
            return flags
        
        # Look for currency context in text
        text_lower = text.lower()
        inr_context = any(word in text_lower for word in ['rupee', 'inr', 'indian'])
        usd_context = any(word in text_lower for word in ['dollar', 'usd', 'american'])
        
        for money in money_claims:
            money_text = money['text']
            if '$' in money_text and inr_context:
                flags.append("currency_mismatch_usd_inr_context")
            elif '₹' in money_text and usd_context:
                flags.append("currency_mismatch_inr_usd_context")
        
        return flags
    
    def _check_unit_absurdity(self, numbers: List[Dict], text: str) -> List[str]:
        """Check for absurd unit values"""
        flags = []
        
        if not self.rules['unit_absurdity']['enabled']:
            return flags
        
        text_lower = text.lower()
        
        for number in numbers:
            try:
                # Extract numeric value
                num_text = re.sub(r'[,\s]', '', number['text'])
                num_text = re.sub(r'(billion|million|thousand|B|M|K)', '', num_text, flags=re.IGNORECASE)
                value = float(num_text)
                
                # Apply multipliers
                if re.search(r'billion|B\b', number['text'], re.IGNORECASE):
                    value *= 1e9
                elif re.search(r'million|M\b', number['text'], re.IGNORECASE):
                    value *= 1e6
                elif re.search(r'thousand|K\b', number['text'], re.IGNORECASE):
                    value *= 1e3
                
                # Context-based absurdity checks
                if ('height' in text_lower or 'tall' in text_lower) and value > self.thresholds['human_height_cm']:
                    flags.append(f"absurd_height_{value}")
                
                if ('temperature' in text_lower or 'temp' in text_lower) and value > self.thresholds['temperature_celsius']:
                    flags.append(f"absurd_temperature_{value}")
                
                if ('market cap' in text_lower or 'valuation' in text_lower) and value > self.thresholds['market_cap_billion'] * 1e9:
                    flags.append(f"absurd_market_cap_{value/1e9:.1f}B")
            
            except (ValueError, AttributeError):
                continue
        
        return flags
    
    def _check_temporal_conflicts(self, dates: List[Dict], text: str) -> List[str]:
        """Check for temporal inconsistencies"""
        flags = []
        
        if not self.rules['future_past_conflict']['enabled'] or len(dates) < 2:
            return flags
        
        current_year = datetime.now().year
        parsed_dates = []
        
        for date in dates:
            if date['parsed']:
                parsed_date = dateparser.parse(date['parsed'])
                if parsed_date:
                    parsed_dates.append((date['text'], parsed_date))
        
        # Check for future dates conflicting with past anchors
        text_lower = text.lower()
        has_past_anchor = any(phrase in text_lower for phrase in ['as of', 'by', 'until', 'through'])
        
        for date_text, parsed_date in parsed_dates:
            if parsed_date.year > current_year + 10:  # Far future
                if has_past_anchor:
                    flags.append(f"future_past_conflict_{date_text}")
            elif parsed_date.year < 1900:  # Suspiciously old
                flags.append(f"suspicious_old_date_{date_text}")
        
        return flags


def demo():
    """Demo the sanity checker"""
    checker = SanityChecker()
    
    test_sentences = [
        {
            'text': "Stock jumped 500% in one day after the announcement.",
            'claims': {
                'percents': [{'text': '500%', 'start': 12, 'end': 16}],
                'dates': [{'text': 'one day', 'start': 20, 'end': 27, 'parsed': '2024-01-01T00:00:00'}],
                'money': [],
                'numbers': []
            }
        },
        {
            'text': "The CEO is 350 cm tall and weighs $50 million.",
            'claims': {
                'percents': [],
                'dates': [],
                'money': [{'text': '$50 million', 'start': 35, 'end': 46}],
                'numbers': [{'text': '350', 'start': 11, 'end': 14}]
            }
        },
        {
            'text': "Temperature reached 2000°C during normal operation.",
            'claims': {
                'percents': [],
                'dates': [],
                'money': [],
                'numbers': [{'text': '2000', 'start': 20, 'end': 24}]
            }
        }
    ]
    
    print("Sanity Check Analysis:")
    print("-" * 50)
    
    for sent in test_sentences:
        score, flags = checker.check_sentence_claims(sent)
        
        print(f"\nSentence: {sent['text']}")
        print(f"Sanity Score: {score:.3f}")
        print(f"Flags: {flags}")


if __name__ == "__main__":
    demo()