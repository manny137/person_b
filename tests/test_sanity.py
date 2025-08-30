# tests/test_sanity.py
import pytest
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sanity import SanityChecker

def test_percent_jump_flag():
    """Test extreme percentage jump detection"""
    checker = SanityChecker()

    sentence_data = {
        'text': "Stock jumped 500% in one day.",
        'claims': {
            'percents': [{'text': '500%', 'start': 12, 'end': 16}],
            'dates': [{'text': 'one day', 'start': 20, 'end': 27, 'parsed': '2024-01-01'}],
            'money': [],
            'numbers': []
        }
    }

    score, flags = checker.check_sentence_claims(sentence_data)
    assert score > 0  # Should have sanity issues
    assert any('percent_jump' in flag for flag in flags)

def test_percent_jump_daily_context():
    """Test extreme percentage jump detection with 'daily' context"""
    checker = SanityChecker()

    sentence_data = {
        'text': "The company reported a daily increase of 400%.",
        'claims': {
            'percents': [{'text': '400%', 'start': 36, 'end': 40}],
            'dates': [],
            'money': [],
            'numbers': []
        }
    }

    score, flags = checker.check_sentence_claims(sentence_data)
    assert score > 0
    assert any('percent_jump' in flag for flag in flags)

def test_temporal_conflict_future_date():
    """Test future date with past anchor"""
    checker = SanityChecker()

    sentence_data = {
        'text': "As of 2020, the company expects to release the product in 2028.",
        'claims': {
            'percents': [],
            'dates': [
                {'text': '2020', 'start': 7, 'end': 11, 'parsed': '2020-01-01T00:00:00'},
                {'text': '2028', 'start': 58, 'end': 62, 'parsed': '2028-01-01T00:00:00'}
            ],
            'money': [],
            'numbers': []
        }
    }

    score, flags = checker.check_sentence_claims(sentence_data)
    assert score > 0
    assert any('future_past_conflict' in flag for flag in flags)

def test_currency_mismatch():
    """Test currency context mismatch detection"""
    checker = SanityChecker()

    sentence_data = {
        'text': "The Indian company reported $100 million in rupee revenue.",
        'claims': {
            'percents': [],
            'dates': [],
            'money': [{'text': '$100 million', 'start': 25, 'end': 37}],
            'numbers': []
        }
    }

    score, flags = checker.check_sentence_claims(sentence_data)
    assert any('currency_mismatch' in flag for flag in flags)

def test_absurd_units():
    """Test absurd unit detection"""
    checker = SanityChecker()

    sentence_data = {
        'text': "The CEO is 400 cm tall.",
        'claims': {
            'percents': [],
            'dates': [],
            'money': [],
            'numbers': [{'text': '400', 'start': 11, 'end': 14}]
        }
    }

    score, flags = checker.check_sentence_claims(sentence_data)
    assert any('absurd_height' in flag for flag in flags)

def test_normal_claims():
    """Test that normal claims don't trigger flags"""
    checker = SanityChecker()

    sentence_data = {
        'text': "Revenue grew 15% last quarter to $500 million.",
        'claims': {
            'percents': [{'text': '15%', 'start': 13, 'end': 16}],
            'dates': [],
            'money': [{'text': '$500 million', 'start': 34, 'end': 46}],
            'numbers': []
        }
    }

    score, flags = checker.check_sentence_claims(sentence_data)
    assert score == 0.0  # Should be clean
    assert len(flags) == 0
