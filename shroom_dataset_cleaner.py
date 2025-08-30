#!/usr/bin/env python3
"""
SHROOM Dataset Cleaner
Clean and standardize SHROOM dataset to prevent regex errors
"""

import json
import re
import unicodedata
from pathlib import Path
from typing import Dict, List, Any, Optional
import shutil

class SHROOMDatasetCleaner:
    """Clean SHROOM dataset for regex compatibility"""
    
    def __init__(self):
        self.cleaning_stats = {
            'files_processed': 0,
            'records_cleaned': 0,
            'issues_fixed': []
        }
    
    def clean_all_datasets(self, backup: bool = True) -> Dict[str, Any]:
        """Clean all SHROOM datasets"""
        print("SHROOM Dataset Cleaning")
        print("=" * 40)
        
        # File mapping based on your actual structure
        files_to_clean = [
            "data/SHROOM_dev-v2/val.model-agnostic.json",
            "data/SHROOM_dev-v2/val.model-aware.v2.json",  # Note: .v2 extension
            "data/SHROOM_test-labeled/test.model-agnostic.json", 
            "data/SHROOM_test-labeled/test.model-aware.json"
        ]
        
        for file_path in files_to_clean:
            if Path(file_path).exists():
                print(f"\nProcessing: {file_path}")
                self.clean_single_file(file_path, backup=backup)
            else:
                print(f"⚠️  File not found: {file_path}")
        
        # Create standardized file names if needed
        self._standardize_filenames()
        
        return self.cleaning_stats
    
    def clean_single_file(self, file_path: str, backup: bool = True):
        """Clean a single SHROOM JSON file"""
        file_path_obj = Path(file_path)
        
        # Create backup
        if backup:
            backup_path = file_path_obj.with_suffix('.json.backup')
            if not backup_path.exists():
                shutil.copy2(file_path_obj, backup_path)
                print(f"  📁 Backup created: {backup_path.name}")
        
        # Load data
        try:
            with open(file_path_obj, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            print(f"  ❌ Error loading {file_path}: {e}")
            return
        
        if not isinstance(data, list):
            print(f"  ⚠️  Expected list, got {type(data)}")
            return
        
        print(f"  📊 Original records: {len(data)}")
        
        # Clean each record
        cleaned_data = []
        records_with_issues = 0
        
        for i, record in enumerate(data):
            if i % 500 == 0 and i > 0:
                print(f"    Processed {i}/{len(data)} records...")
            
            cleaned_record, had_issues = self.clean_record(record, i)
            if had_issues:
                records_with_issues += 1
            cleaned_data.append(cleaned_record)
        
        # Save cleaned data
        try:
            with open(file_path_obj, 'w', encoding='utf-8') as f:
                json.dump(cleaned_data, f, indent=2, ensure_ascii=False)
            
            print(f"  ✅ Cleaned {records_with_issues} records with issues")
            print(f"  💾 Saved cleaned data to {file_path}")
            
            self.cleaning_stats['files_processed'] += 1
            self.cleaning_stats['records_cleaned'] += records_with_issues
            
        except Exception as e:
            print(f"  ❌ Error saving {file_path}: {e}")
    
    def clean_record(self, record: Dict[str, Any], record_idx: int) -> tuple[Dict[str, Any], bool]:
        """Clean individual record"""
        cleaned_record = record.copy()
        had_issues = False
        
        # Text fields to clean
        text_fields = ['hyp', 'src', 'tgt', 'ref']
        
        for field in text_fields:
            if field in cleaned_record:
                original_text = cleaned_record[field]
                if isinstance(original_text, str):
                    cleaned_text, field_had_issues = self.clean_text(original_text, record_idx, field)
                    if field_had_issues:
                        had_issues = True
                        cleaned_record[field] = cleaned_text
        
        # Clean labels if they're strings
        if 'labels' in cleaned_record and isinstance(cleaned_record['labels'], list):
            cleaned_labels = []
            for label in cleaned_record['labels']:
                if isinstance(label, str):
                    cleaned_label, _ = self.clean_text(label, record_idx, 'labels')
                    cleaned_labels.append(cleaned_label)
                else:
                    cleaned_labels.append(label)
            cleaned_record['labels'] = cleaned_labels
        
        if 'label' in cleaned_record and isinstance(cleaned_record['label'], str):
            cleaned_label, label_had_issues = self.clean_text(cleaned_record['label'], record_idx, 'label')
            if label_had_issues:
                had_issues = True
                cleaned_record['label'] = cleaned_label
        
        return cleaned_record, had_issues
    
    def clean_text(self, text: str, record_idx: int, field: str) -> tuple[str, bool]:
        """Clean individual text field"""
        if not isinstance(text, str):
            return str(text), False
        
        original_text = text
        had_issues = False
        
        # 1. Handle Unicode normalization
        text = unicodedata.normalize('NFKC', text)
        
        # 2. Remove or escape problematic regex patterns
        # Escape backslash followed by digits (regex backreferences)
        if re.search(r'\\[0-9]+', text):
            text = re.sub(r'\\([0-9]+)', r'\\\\\\1', text)
            had_issues = True
            self.cleaning_stats['issues_fixed'].append(f"Record {record_idx} {field}: Escaped backslash references")
        
        # Escape dollar followed by digits (group references)  
        if re.search(r'\$[0-9]+', text):
            text = re.sub(r'\$([0-9]+)', r'\\$\\1', text)
            had_issues = True
            self.cleaning_stats['issues_fixed'].append(f"Record {record_idx} {field}: Escaped dollar references")
        
        # 3. Clean whitespace
        # Replace multiple whitespace with single space
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        # 4. Handle null bytes and control characters
        text = text.replace('\x00', '')
        text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)
        
        # 5. Escape other potentially problematic regex special chars if excessive
        special_chars_to_check = ['^', '|', '*', '+', '?']
        for char in special_chars_to_check:
            count = text.count(char)
            if count > 20:  # If excessive, likely not intended as regex
                text = text.replace(char, '\\' + char)
                had_issues = True
                self.cleaning_stats['issues_fixed'].append(f"Record {record_idx} {field}: Escaped excessive '{char}' chars")
        
        # 6. Fix unbalanced parentheses (common cause of regex issues)
        open_parens = text.count('(')
        close_parens = text.count(')')
        if open_parens != close_parens:
            # Simple fix: escape all parentheses if unbalanced
            text = text.replace('(', '\\(').replace(')', '\\)')
            had_issues = True
            self.cleaning_stats['issues_fixed'].append(f"Record {record_idx} {field}: Fixed unbalanced parentheses")
        
        return text, had_issues
    
    def _standardize_filenames(self):
        """Create standardized filenames for compatibility"""
        # Create standard name for the .v2 file
        v2_file = Path("data/SHROOM_dev-v2/val.model-aware.v2.json")
        standard_file = Path("data/SHROOM_dev-v2/val.model-aware.json")
        
        if v2_file.exists() and not standard_file.exists():
            shutil.copy2(v2_file, standard_file)
            print(f"  🔄 Created standardized filename: {standard_file.name}")
            self.cleaning_stats['issues_fixed'].append("Created standardized filename for model-aware file")
    
    def validate_cleaned_data(self) -> Dict[str, Any]:
        """Validate that cleaned data doesn't have regex issues"""
        print("\nValidating cleaned data...")
        
        validation_results = {
            'files_checked': 0,
            'potential_issues': [],
            'all_clear': True
        }
        
        files_to_check = [
            "data/SHROOM_dev-v2/val.model-agnostic.json",
            "data/SHROOM_dev-v2/val.model-aware.json",
            "data/SHROOM_test-labeled/test.model-agnostic.json", 
            "data/SHROOM_test-labeled/test.model-aware.json"
        ]
        
        for file_path in files_to_check:
            if Path(file_path).exists():
                validation_results['files_checked'] += 1
                issues = self._check_file_for_regex_issues(file_path)
                if issues:
                    validation_results['potential_issues'].extend(issues)
                    validation_results['all_clear'] = False
        
        if validation_results['all_clear']:
            print("✅ All files validated - no regex issues detected")
        else:
            print(f"⚠️  Found {len(validation_results['potential_issues'])} potential issues")
            for issue in validation_results['potential_issues'][:5]:  # Show first 5
                print(f"  {issue}")
        
        return validation_results
    
    def _check_file_for_regex_issues(self, file_path: str) -> List[str]:
        """Check file for remaining regex issues"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except:
            return [f"Could not load {file_path}"]
        
        issues = []
        text_fields = ['hyp', 'src', 'tgt', 'ref', 'label']
        
        for i, record in enumerate(data[:20]):  # Check first 20 records
            for field in text_fields:
                if field in record and isinstance(record[field], str):
                    text = record[field]
                    # Check for unescaped regex patterns
                    if re.search(r'(?<!\\)\\[0-9]+', text):
                        issues.append(f"{Path(file_path).name} record {i}: Unescaped backslash reference in {field}")
                    if re.search(r'(?<!\\)\$[0-9]+', text):
                        issues.append(f"{Path(file_path).name} record {i}: Unescaped dollar reference in {field}")
        
        return issues
    
    def generate_cleaning_report(self) -> str:
        """Generate cleaning report"""
        report = [
            "SHROOM Dataset Cleaning Report",
            "=" * 50,
            f"Files processed: {self.cleaning_stats['files_processed']}",
            f"Records with issues fixed: {self.cleaning_stats['records_cleaned']}",
            f"Total fixes applied: {len(self.cleaning_stats['issues_fixed'])}",
            "",
            "Issues Fixed:",
            "-" * 20
        ]
        
        # Group similar issues
        issue_counts = {}
        for issue in self.cleaning_stats['issues_fixed']:
            issue_type = issue.split(':')[1].strip() if ':' in issue else issue
            issue_counts[issue_type] = issue_counts.get(issue_type, 0) + 1
        
        for issue_type, count in sorted(issue_counts.items(), key=lambda x: x[1], reverse=True):
            report.append(f"  {issue_type}: {count} instances")
        
        return "\n".join(report)


def main():
    """Run the SHROOM dataset cleaner"""
    cleaner = SHROOMDatasetCleaner()
    
    # Clean all datasets
    stats = cleaner.clean_all_datasets(backup=True)
    
    # Validate results
    validation = cleaner.validate_cleaned_data()
    
    # Generate report
    report = cleaner.generate_cleaning_report()
    print(f"\n{report}")
    
    # Save report
    with open("cleaning_report.txt", "w") as f:
        f.write(report)
    print("\n📄 Cleaning report saved to: cleaning_report.txt")
    
    print("\nNext steps:")
    print("1. Test your pipeline with the cleaned data")
    print("2. If you still get regex errors, check your pipeline code")
    print("3. Backup files are saved as *.backup in case you need to revert")


if __name__ == "__main__":
    main()