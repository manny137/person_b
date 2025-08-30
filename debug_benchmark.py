#!/usr/bin/env python3
"""
Debug version of benchmark.py to identify regex error sources
"""

import json
import traceback
import re
from pathlib import Path
from typing import Dict, List, Any, Tuple

from data_loader import SHROOMLoader
from pipeline import ClaimsPipeline


class RegexErrorDebugger:
    """Debug regex errors in the pipeline"""
    
    def __init__(self):
        self.loader = SHROOMLoader()
        self.pipeline = ClaimsPipeline()
        self.error_log = []
    
    def debug_pipeline_processing(self, max_samples: int = 200) -> Dict[str, Any]:
        """Debug pipeline processing to find regex errors"""
        
        print("REGEX ERROR DEBUGGING")
        print("=" * 30)
        
        # Get some sample data
        available = self.loader.get_available_splits()
        if not available.get('splits'):
            print("No data splits available")
            return {'error': 'No data available'}
        
        split = available['splits'][0]
        model_type = available['model_types'][0] if available['model_types'] else 'model-agnostic'
        
        print(f"Testing on {split} split ({model_type})")
        texts_labels = self.loader.extract_hypothesis_texts(split, model_type)
        
        if max_samples:
            texts_labels = texts_labels[:max_samples]
        
        print(f"Processing {len(texts_labels)} texts to find regex errors...")
        
        debug_results = {
            'total_texts': len(texts_labels),
            'successful_processing': 0,
            'regex_errors': [],
            'other_errors': [],
            'problematic_texts': []
        }
        
        for i, (text, label) in enumerate(texts_labels):
            if i % 50 == 0:
                print(f"Processed {i}/{len(texts_labels)} texts...")
            
            try:
                # Test the problematic text first
                self._check_text_for_regex_issues(text, i)
                
                # Try processing with pipeline
                result = self.pipeline.process_text(text)
                debug_results['successful_processing'] += 1
                
            except Exception as e:
                error_str = str(e)
                error_info = {
                    'text_index': i,
                    'text': text,
                    'label': label,
                    'error': error_str,
                    'error_type': type(e).__name__,
                    'traceback': traceback.format_exc()
                }
                
                if 'invalid group reference' in error_str:
                    debug_results['regex_errors'].append(error_info)
                    print(f"\n🚨 REGEX ERROR at text {i}:")
                    print(f"Text: {text[:200]}...")
                    print(f"Error: {error_str}")
                    
                    # Deep analysis of this specific text
                    self._deep_analyze_problematic_text(text, i)
                    
                else:
                    debug_results['other_errors'].append(error_info)
        
        # Summary
        print(f"\n📊 DEBUGGING SUMMARY:")
        print(f"Successful processing: {debug_results['successful_processing']}")
        print(f"Regex errors: {len(debug_results['regex_errors'])}")
        print(f"Other errors: {len(debug_results['other_errors'])}")
        
        if debug_results['regex_errors']:
            print(f"\n🔍 REGEX ERROR ANALYSIS:")
            self._analyze_regex_errors(debug_results['regex_errors'])
        
        return debug_results
    
    def _check_text_for_regex_issues(self, text: str, text_index: int):
        """Pre-check text for potential regex issues"""
        if not isinstance(text, str):
            return
        
        # Check for obvious regex backreference patterns
        patterns_to_check = [
            r'\\[0-9]+',      # Backslash followed by digits
            r'\$[0-9]+',      # Dollar followed by digits  
            r'\\g<[0-9]+>',   # Named group references
            r'\([^)]*\$[0-9]+[^)]*\)',  # Dollar refs inside parentheses
        ]
        
        for pattern in patterns_to_check:
            matches = re.findall(pattern, text)
            if matches:
                print(f"⚠️  Text {text_index} contains potential regex pattern: {pattern} -> {matches}")
    
    def _deep_analyze_problematic_text(self, text: str, text_index: int):
        """Deep analysis of text causing regex errors"""
        print(f"\n🔬 DEEP ANALYSIS of text {text_index}:")
        print(f"Text length: {len(text)}")
        print(f"Full text: {repr(text)}")  # Shows escape sequences
        
        # Check for specific patterns
        problematic_patterns = {
            'backslash_digits': re.findall(r'\\[0-9]+', text),
            'dollar_digits': re.findall(r'\$[0-9]+', text),
            'unescaped_backslash': text.count('\\'),
            'dollar_signs': text.count('$'),
            'parentheses': f"{text.count('(')} open, {text.count(')')} close",
            'brackets': f"{text.count('[')} open, {text.count(']')} close",
            'curly_braces': f"{text.count('{')} open, {text.count('}')} close"
        }
        
        print("Problematic patterns found:")
        for pattern_name, result in problematic_patterns.items():
            if result:
                print(f"  {pattern_name}: {result}")
        
        # Try to isolate where in the pipeline the error occurs
        self._test_pipeline_components(text, text_index)
    
    def _test_pipeline_components(self, text: str, text_index: int):
        """Test individual pipeline components to isolate the error"""
        print(f"\n🧪 TESTING PIPELINE COMPONENTS for text {text_index}:")
        
        # Test basic text operations first
        try:
            # Test if it's a string operation issue
            test_replace = text.replace("$", "DOLLAR")
            print("✅ Basic string replace works")
        except Exception as e:
            print(f"❌ Basic string replace fails: {e}")
        
        try:
            # Test regex operations
            test_regex = re.search(r'test', text)
            print("✅ Basic regex search works")
        except Exception as e:
            print(f"❌ Basic regex search fails: {e}")
        
        try:
            # Test regex substitute
            test_sub = re.sub(r'test', 'TEST', text)
            print("✅ Basic regex substitute works")
        except Exception as e:
            print(f"❌ Basic regex substitute fails: {e}")
            
        try:
            # Test more complex substitution that might trigger the error
            # This is likely where the "invalid group reference 3" comes from
            test_complex_sub = re.sub(r'(\w+)', r'\1', text)  # Simple backreference
            print("✅ Simple backreference substitute works")
        except Exception as e:
            print(f"❌ Simple backreference substitute fails: {e}")
            
        # Test if the error is in a specific substitution pattern
        problematic_substitutions = [
            (r'(\w+)(\s+)(\w+)', r'\1\2\3'),  # 3-group pattern - this might be the issue!
            (r'(\d+)', r'\1'),
            (r'(\w+)', r'[\1]'),
        ]
        
        for pattern, repl in problematic_substitutions:
            try:
                result = re.sub(pattern, repl, text)
                print(f"✅ Pattern {pattern} -> {repl} works")
            except Exception as e:
                print(f"❌ Pattern {pattern} -> {repl} FAILS: {e}")
                print(f"   This is likely the source of your 'invalid group reference 3' error!")
    
    def _analyze_regex_errors(self, regex_errors: List[Dict]):
        """Analyze patterns in regex errors"""
        print("\nCommon patterns in regex errors:")
        
        # Look for common substrings in problematic texts
        error_texts = [err['text'] for err in regex_errors[:5]]  # First 5 errors
        
        for i, text in enumerate(error_texts):
            print(f"\nError text {i + 1}:")
            print(f"  Length: {len(text)}")
            print(f"  Preview: {text[:100]}...")
            print(f"  Contains backslashes: {text.count('\\')}")
            print(f"  Contains dollars: {text.count('$')}")
            
            # Look for specific problem patterns
            if re.search(r'\$[0-9]', text):
                matches = re.findall(r'\$[0-9]+', text)
                print(f"  ⚠️  Dollar+digit patterns: {matches}")
            
            if re.search(r'\\[0-9]', text):
                matches = re.findall(r'\\[0-9]+', text)
                print(f"  ⚠️  Backslash+digit patterns: {matches}")
    
    def fix_problematic_texts_in_pipeline(self) -> str:
        """Generate code to fix the pipeline"""
        fix_code = '''
# Add this to your pipeline.py to escape problematic regex patterns:

def escape_regex_patterns(text):
    """Escape patterns that cause 'invalid group reference' errors"""
    if not isinstance(text, str):
        return str(text)
    
    # Escape dollar followed by digits (group references)
    text = re.sub(r'\\$([0-9]+)', r'\\\\$\\1', text)
    
    # Escape backslash followed by digits (backreferences)  
    text = re.sub(r'\\\\([0-9]+)', r'\\\\\\\\\\1', text)
    
    return text

# Use this at the start of process_text():
def process_text(self, text):
    text = escape_regex_patterns(text)  # Add this line
    # ... rest of your existing code
        '''
        return fix_code


def main():
    """Run debugging"""
    debugger = RegexErrorDebugger()
    
    print("Starting regex error debugging...")
    results = debugger.debug_pipeline_processing(max_samples=300)
    
    # Save debug results
    with open("regex_debug_results.json", 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nDebug results saved to: regex_debug_results.json")
    
    if results.get('regex_errors'):
        print("\n🔧 SUGGESTED FIX:")
        fix_code = debugger.fix_problematic_texts_in_pipeline()
        print(fix_code)
        
        with open("pipeline_fix.py", 'w') as f:
            f.write(fix_code)
        print("Fix code saved to: pipeline_fix.py")
    
    print("\nDebugging complete!")


if __name__ == "__main__":
    main()