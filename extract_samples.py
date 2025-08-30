#!/usr/bin/env python3
"""
Extract sample data from SHROOM dataset for debugging
Based on the file structure shown
"""

import json
import os
from pathlib import Path

def extract_shroom_samples():
    """Extract sample data from SHROOM files"""
    
    # Base paths - matching your file structure
    base_paths = [
        "data/SHROOM_dev-v2",
        "data/SHROOM_test-labeled", 
        "data/SHROOM_unlabeled"
    ]
    
    print("SHROOM Dataset Sample Extraction")
    print("=" * 40)
    
    for base_path in base_paths:
        base_path_obj = Path(base_path)
        
        if not base_path_obj.exists():
            print(f"Path not found: {base_path}")
            continue
            
        print(f"\nExploring: {base_path}")
        print("-" * 30)
        
        # Look for JSON files in this directory
        json_files = list(base_path_obj.glob("*.json"))
        
        for json_file in json_files[:2]:  # First 2 files per directory
            print(f"\nFile: {json_file.name}")
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                print(f"  Type: {type(data)}")
                if isinstance(data, list):
                    print(f"  Length: {len(data)} records")
                    if data:
                        print("  Sample record structure:")
                        sample = data[0]
                        print(f"    Keys: {list(sample.keys()) if isinstance(sample, dict) else 'Not a dict'}")
                        
                        # Show sample content (truncated)
                        print("  First record sample:")
                        sample_str = json.dumps(sample, indent=4)[:500]
                        print("   ", sample_str.replace('\n', '\n    '))
                        if len(json.dumps(sample)) > 500:
                            print("    ... (truncated)")
                        
                        # Look for problematic patterns
                        print("  Checking for potential regex issues...")
                        text_fields = ['hyp', 'hypothesis', 'text', 'src', 'ref', 'tgt']
                        for field in text_fields:
                            if field in sample:
                                text = str(sample[field])
                                issues = check_text_for_regex_issues(text, field)
                                if issues:
                                    print(f"    ⚠️  {field}: {issues}")
                                else:
                                    print(f"    ✅ {field}: OK")
                        
                        # Show more samples if they exist
                        if len(data) > 1:
                            print(f"  \nSample of next few records:")
                            for i in range(1, min(4, len(data))):
                                record = data[i]
                                for field in text_fields:
                                    if field in record:
                                        text = str(record[field])[:100]
                                        print(f"    Record {i} {field}: {text}...")
                                        break
                
                elif isinstance(data, dict):
                    print(f"  Keys: {list(data.keys())}")
                    print("  Sample content:")
                    sample_str = json.dumps(data, indent=4)[:500]
                    print("   ", sample_str.replace('\n', '\n    '))
                    
            except Exception as e:
                print(f"  Error reading {json_file.name}: {e}")

def check_text_for_regex_issues(text, field_name):
    """Check text for patterns that might cause regex errors"""
    issues = []
    
    if not isinstance(text, str):
        issues.append(f"Not a string: {type(text)}")
        return issues
    
    # Check for group references that might cause "invalid group reference" errors
    import re
    
    # Look for potential backreferences
    if re.search(r'\\[0-9]+', text):
        issues.append("Contains backslash + digits (\\1, \\2, etc.)")
    
    # Look for dollar references  
    if re.search(r'\$[0-9]+', text):
        issues.append("Contains dollar + digits ($1, $2, etc.)")
    
    # Look for excessive regex special characters
    special_chars = ['\\', '$', '^', '|', '?', '*', '+', '{', '}', '(', ')', '[', ']']
    for char in special_chars:
        count = text.count(char)
        if count > 10:
            issues.append(f"Many '{char}' characters ({count})")
    
    # Check for unbalanced parentheses (common in regex)
    if text.count('(') != text.count(')'):
        issues.append(f"Unbalanced parentheses: {text.count('(')} open, {text.count(')')} close")
    
    # Look for null bytes
    if '\x00' in text:
        issues.append("Contains null bytes")
    
    return issues

def find_specific_files():
    """Look for the specific files mentioned in the error"""
    possible_locations = [
        "data/SHROOM_dev-v2/val.model-agnostic.json",
        "data/SHROOM_dev-v2/val.model-aware.json", 
        "data/SHROOM_test-labeled/test.model-agnostic.json",
        "data/SHROOM_test-labeled/test.model-aware.json",
        "data/SHROOM_unlabeled/train.model-agnostic.json",
        "data/SHROOM_unlabeled/train.model-aware.json"
    ]
    
    print("\nLooking for specific files that might be causing issues:")
    print("-" * 50)
    
    for file_path in possible_locations:
        file_path_obj = Path(file_path)
        if file_path_obj.exists():
            print(f"✅ Found: {file_path}")
            
            # Quick check of this file
            try:
                with open(file_path_obj, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    print(f"   Records: {len(data) if isinstance(data, list) else 'Dict structure'}")
                    
                    # Check first few records for issues
                    if isinstance(data, list) and data:
                        problematic_records = []
                        for i, record in enumerate(data[:10]):  # Check first 10
                            text_fields = ['hyp', 'hypothesis', 'text', 'src', 'ref', 'tgt']
                            for field in text_fields:
                                if field in record:
                                    text = str(record[field])
                                    issues = check_text_for_regex_issues(text, field)
                                    if issues:
                                        problematic_records.append(f"Record {i}: {field} - {issues[0]}")
                        
                        if problematic_records:
                            print("   ⚠️  Potential issues found:")
                            for issue in problematic_records[:3]:  # Show first 3 issues
                                print(f"      {issue}")
                            if len(problematic_records) > 3:
                                print(f"      ... and {len(problematic_records) - 3} more")
                        else:
                            print("   ✅ No obvious regex issues in first 10 records")
                            
            except Exception as e:
                print(f"   ❌ Error reading file: {e}")
        else:
            print(f"❌ Not found: {file_path}")

if __name__ == "__main__":
    extract_shroom_samples()
    find_specific_files()
    
    print("\n" + "=" * 50)
    print("Next steps:")
    print("1. Review the sample data above")
    print("2. Look for any ⚠️  warnings about regex issues")
    print("3. If issues found, we can create a cleaning script")
    print("4. Share this output to help debug the 'invalid group reference 3' error")