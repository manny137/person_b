# tests/test_parser.py
import pytest
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from parser import ClaimExtractor

def test_claim_extractor_entities():
    """Test entity extraction"""
    extractor = ClaimExtractor()
    text = "Apple Inc. and Microsoft Corp. are competing in the technology sector."
    
    sentences = extractor.extract_sentence_claims(text)
    assert len(sentences) == 1
    
    entities = sentences[0]['claims']['entities']
    entity_texts = [e['text'] for e in entities]
    assert 'Apple Inc.' in entity_texts
    assert 'Microsoft Corp.' in entity_texts

def test_claim_extractor_numbers():
    """Test number and percent extraction"""
    extractor = ClaimExtractor()
    text = "Revenue increased by 25.5% to $1.2 billion."
    
    sentences = extractor.extract_sentence_claims(text)
    claims = sentences[0]['claims']
    
    assert len(claims['percents']) == 1
    assert claims['percents'][0]['text'] == '25.5%'
    
    assert len(claims['money']) == 1
    assert '$1.2 billion' in claims['money'][0]['text']

def test_claim_extractor_tickers():
    """Test ticker extraction"""
    extractor = ClaimExtractor()
    text = "AAPL and MSFT stocks are up today."
    
    sentences = extractor.extract_sentence_claims(text)
    claims = sentences[0]['claims']
    
    tickers = [t['text'] for t in claims['tickers']]
    assert 'AAPL' in tickers
    assert 'MSFT' in tickers


# tests/test_speculative.py
import pytest
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from speculative import SpeculativeScorer

def test_hedge_detection():
    """Test hedge word detection"""
    scorer = SpeculativeScorer()
    
    # Test hedge words
    score, counts = scorer.score_sentence("The company might possibly see growth.")
    assert counts['hedges'] >= 2  # 'might' and 'possibly'
    assert score > 0
    
def test_absolute_detection():
    """Test absolute word detection"""
    scorer = SpeculativeScorer()
    
    score, counts = scorer.score_sentence("This is definitely guaranteed to work.")
    assert counts['absolutes'] >= 2  # 'definitely' and 'guaranteed'
    assert score > 0

def test_neutral_sentence():
    """Test neutral sentence with no speculative language"""
    scorer = SpeculativeScorer()
    
    score, counts = scorer.score_sentence("The company reported revenue of $100 million.")
    assert counts['hedges'] == 0
    assert counts['absolutes'] == 0
    assert score == 0.0

def test_matched_words():
    """Test word matching functionality"""
    scorer = SpeculativeScorer()
    
    matches = scorer.get_matched_words("This might possibly work.")
    assert 'might' in matches['hedges'] 
    assert 'possibly' in matches['hedges']


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


# tests/test_paraphrase.py
import pytest
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from paraphrase import ParaphraseGenerator

def test_paraphrase_generation():
    """Test that 3 paraphrases are generated"""
    generator = ParaphraseGenerator()
    
    text = "The company reported strong earnings that increased significantly."
    paraphrases = generator.generate_paraphrases(text)
    
    assert len(paraphrases) == 3
    assert all(isinstance(p, str) for p in paraphrases)
    assert all(len(p.strip()) > 0 for p in paraphrases)

def test_synonym_substitution():
    """Test synonym substitution paraphrase"""
    generator = ParaphraseGenerator()
    
    text = "The company will increase revenue."
    paraphrase = generator._synonym_paraphrase(text)
    
    # Should replace 'increase' with 'rise' and 'company' with 'firm'
    assert 'rise' in paraphrase.lower() or 'firm' in paraphrase.lower()

def test_instability_calculation():
    """Test instability score calculation"""
    generator = ParaphraseGenerator()
    
    # Test with varying scores (high instability)
    original = {'spec_score': 0.5, 'num_sanity_score': 0.1}
    paraphrases = [
        {'spec_score': 0.8, 'num_sanity_score': 0.3},
        {'spec_score': 0.2, 'num_sanity_score': 0.0},
        {'spec_score': 0.6, 'num_sanity_score': 0.2}
    ]
    
    instability = generator.calculate_instability(original, paraphrases)
    assert instability > 0  # Should show instability
    
    # Test with consistent scores (low instability)
    consistent_paraphrases = [
        {'spec_score': 0.5, 'num_sanity_score': 0.1},
        {'spec_score': 0.5, 'num_sanity_score': 0.1},
        {'spec_score': 0.5, 'num_sanity_score': 0.1}
    ]
    
    stable_instability = generator.calculate_instability(original, consistent_paraphrases)
    assert stable_instability == 0.0  # Should be perfectly stable

def test_deterministic_output():
    """Test that paraphrases are deterministic"""
    generator1 = ParaphraseGenerator()
    generator2 = ParaphraseGenerator()
    
    text = "Apple reported strong quarterly results."
    
    paraphrases1 = generator1.generate_paraphrases(text)
    paraphrases2 = generator2.generate_paraphrases(text)
    
    assert paraphrases1 == paraphrases2  # Should be identical


if __name__ == "__main__":
    # Run all tests
    import subprocess
    result = subprocess.run(['python', '-m', 'pytest', __file__, '-v'], 
                          capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)