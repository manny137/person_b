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
import spacy


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

        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            raise RuntimeError("Please install spaCy English model")

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

        numeric_claims = (len(claims['percents']) + len(claims['money']) +
                         len(claims['numbers']) + len(claims['dates']))

        if numeric_claims == 0:
            return 0.0, []

        flags.extend(self._check_percent_jumps(claims['percents'], claims['dates'], text))
        flags.extend(self._check_currency_mismatch(claims['money'], text))
        flags.extend(self._check_unit_absurdity(claims['numbers'], text))
        flags.extend(self._check_temporal_conflicts(claims['dates'], text))

        num_sanity_score = len(flags) / max(1, numeric_claims)

        return min(num_sanity_score, 1.0), flags

    def _check_percent_jumps(self, percents: List[Dict], dates: List[Dict], text: str) -> List[str]:
        """Check for unrealistic percentage jumps in short time periods"""
        flags = []
        rule = self.rules['percent_jump']

        if not rule['enabled'] or not percents:
            return flags

        is_daily_context = any(keyword in text.lower() for keyword in ['in one day', 'in a single day', 'daily']) or \
                           any('day' in d['text'].lower() for d in dates)

        if is_daily_context:
            threshold = rule['threshold']
            for percent in percents:
                percent_text = percent['text'].replace('%', '')
                try:
                    value = abs(float(percent_text))
                    if value > threshold:
                        flags.append(f"percent_jump_{value}")
                except ValueError:
                    continue

        return flags

    def _check_currency_mismatch(self, money_claims: List[Dict], text: str) -> List[str]:
        """Check for currency symbol/context mismatches"""
        flags = []
        if not self.rules['currency_mismatch']['enabled']:
            return flags

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
                num_text = re.sub(r'[,\]s]', '', number['text'])
                num_text = re.sub(r'(billion|million|thousand|B|M|K)', '', num_text, flags=re.IGNORECASE)
                value = float(num_text)

                if re.search(r'billion|B\b', number['text'], re.IGNORECASE):
                    value *= 1e9
                elif re.search(r'million|M\b', number['text'], re.IGNORECASE):
                    value *= 1e6
                elif re.search(r'thousand|K\b', number['text'], re.IGNORECASE):
                    value *= 1e3

                # Human height
                if ('height' in text_lower or 'tall' in text_lower) and 'cm' in text_lower:
                    if value > self.thresholds['human_height_cm']:
                        flags.append(f"absurd_height_{value}cm")

                # Human weight
                if ('weight' in text_lower or 'weighs' in text_lower) and 'kg' in text_lower:
                    if value > self.thresholds['human_weight_kg']:
                        flags.append(f"absurd_weight_{value}kg")

                # Vehicle speed
                if ('speed' in text_lower or 'fast' in text_lower or 'driving' in text_lower) and ('kph' in text_lower or 'km/h' in text_lower):
                    if value > self.thresholds['vehicle_speed_kph']:
                        flags.append(f"absurd_speed_{value}kph")

                # Distance
                if ('distance' in text_lower or 'far' in text_lower) and 'km' in text_lower:
                    if value > self.thresholds['distance_km']:
                        flags.append(f"absurd_distance_{value}km")

                # Temperature
                if ('temperature' in text_lower or 'temp' in text_lower) and ('°c' in text_lower or 'celsius' in text_lower):
                    if value > self.thresholds['temperature_celsius']:
                        flags.append(f"absurd_temperature_{value}C")

                # Market cap
                if ('market cap' in text_lower or 'valuation' in text_lower) and value > self.thresholds['market_cap_billion'] * 1e9:
                    flags.append(f"absurd_market_cap_{value/1e9:.1f}B")

            except (ValueError, AttributeError):
                continue

        return list(set(flags))

    def _check_temporal_conflicts(self, dates: List[Dict], text: str) -> List[str]:
        """Check for temporal inconsistencies using verb tense"""
        flags = []
        if not self.rules['future_past_conflict']['enabled'] or not dates:
            return flags

        current_year = datetime.now().year
        doc = self.nlp(text)

        # Check for past tense verbs
        is_past_tense = any(token.tag_ == "VBD" for token in doc)

        if is_past_tense:
            for date in dates:
                if date.get('parsed'):
                    parsed_date = dateparser.parse(date['parsed'])
                    if parsed_date and parsed_date.year > current_year:
                        flags.append(f"future_date_in_past_tense_sentence_{date['text']}")

        # Keep the old checks as well, as they are more explicit
        parsed_dates = []
        for date in dates:
            if date.get('parsed'):
                parsed_date = dateparser.parse(date['parsed'])
                if parsed_date:
                    parsed_dates.append((date['text'], parsed_date))

        if not parsed_dates:
            return list(set(flags))

        text_lower = text.lower()
        has_past_anchor = any(phrase in text_lower for phrase in ['as of', 'by', 'until', 'through', 'reported in'])

        for date_text, parsed_date in parsed_dates:
            if parsed_date.year > current_year:
                if has_past_anchor:
                    flags.append(f"future_past_conflict_{date_text}")

            if parsed_date.year > current_year + 10:
                flags.append(f"far_future_date_{date_text}")
            elif parsed_date.year < 1900:
                flags.append(f"suspicious_old_date_{date_text}")

        return list(set(flags))

def demo():
    """Demo the sanity checker"""
    checker = SanityChecker()

    test_sentences = [
        {
            'text': "Stock jumped 500% in one day after the announcement.",
            'claims': {
                'percents': [{'text': '500%', 'start': 12, 'end': 16}],
                'dates': [{'text': 'one day', 'start': 20, 'end': 27, 'parsed': '2024-01-01T00:00:00'}],
                'money': [], 'numbers': []
            }
        },
        {
            'text': "The CEO is 350 cm tall and weighs 400 kg.",
            'claims': {
                'percents': [], 'dates': [], 'money': [],
                'numbers': [{'text': '350', 'start': 11, 'end': 14}, {'text': '400', 'start': 33, 'end': 36}]
            }
        },
        {
            'text': "The car reached a top speed of 500 kph.",
            'claims': {
                'percents': [], 'dates': [], 'money': [],
                'numbers': [{'text': '500', 'start': 32, 'end': 35}]
            }
        },
        {
            'text': "The company announced it will launch a new product in 2028.",
            'claims': {
                'percents': [], 'money': [], 'numbers': [],
                'dates': [{'text': '2028', 'start': 55, 'end': 59, 'parsed': '2028-01-01T00:00:00'}]
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
