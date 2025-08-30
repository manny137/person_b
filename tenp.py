import difflib
import os
from pathlib import Path

def read_file_content(file_path):
    """Read and return the content of a text file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return None
    except UnicodeDecodeError:
        try:
            with open(file_path, 'r', encoding='latin-1') as file:
                return file.read()
        except Exception as e:
            print(f"Error reading file '{file_path}': {e}")
            return None
    except Exception as e:
        print(f"Error reading file '{file_path}': {e}")
        return None

def compare_texts(text1, text2, file1_name="Text 1", file2_name="Text 2"):
    """Compare two texts and show differences."""
    if text1 is None or text2 is None:
        return
    
    print(f"\n{'='*60}")
    print(f"COMPARISON: {file1_name} vs {file2_name}")
    print(f"{'='*60}")
    
    # Split into lines for better comparison
    lines1 = text1.splitlines(keepends=True)
    lines2 = text2.splitlines(keepends=True)
    
    # Generate unified diff
    diff = list(difflib.unified_diff(
        lines1, lines2,
        fromfile=file1_name,
        tofile=file2_name,
        lineterm=''
    ))
    
    if not diff:
        print("✓ Files are identical!")
        return
    
    print("DIFFERENCES FOUND:")
    print("-" * 40)
    
    for line in diff:
        if line.startswith('+++') or line.startswith('---'):
            print(f"\033[94m{line}\033[0m")  # Blue for file headers
        elif line.startswith('+'):
            print(f"\033[92m{line}\033[0m")  # Green for additions
        elif line.startswith('-'):
            print(f"\033[91m{line}\033[0m")  # Red for deletions
        elif line.startswith('@@'):
            print(f"\033[93m{line}\033[0m")  # Yellow for line numbers
        else:
            print(line)
    
    # Summary statistics
    additions = sum(1 for line in diff if line.startswith('+') and not line.startswith('+++'))
    deletions = sum(1 for line in diff if line.startswith('-') and not line.startswith('---'))
    
    print(f"\n📊 SUMMARY:")
    print(f"   Lines added: {additions}")
    print(f"   Lines removed: {deletions}")
    print(f"   Total changes: {additions + deletions}")

def show_file_content(file_path, max_lines=50):
    """Display the content of a file with line numbers."""
    content = read_file_content(file_path)
    if content is None:
        return
    
    print(f"\n{'='*60}")
    print(f"CONTENT OF: {file_path}")
    print(f"{'='*60}")
    
    lines = content.splitlines()
    total_lines = len(lines)
    
    for i, line in enumerate(lines[:max_lines], 1):
        print(f"{i:4d}: {line}")
    
    if total_lines > max_lines:
        print(f"... ({total_lines - max_lines} more lines)")
    
    print(f"\nTotal lines: {total_lines}")
    print(f"File size: {len(content)} characters")

def find_files_in_directory(directory, pattern="*.txt"):
    """Find files matching a pattern in the directory."""
    path = Path(directory)
    if not path.exists():
        print(f"Directory '{directory}' not found.")
        return []
    
    files = list(path.glob(pattern))
    return sorted(files)

def main():
    """Main function to handle file comparison."""
    print("🔍 TEXT FILE COMPARISON TOOL")
    print("=" * 60)
    
    # Based on your file structure, let's set up some common paths
    base_dirs = [
        "SHROOM_dev-v2",
        "SHROOM_test-labeled",
        "SHROOM_unlabeled"
    ]
    
    print("Available operations:")
    print("1. Compare two specific files")
    print("2. Show content of a single file")
    print("3. List files in directory")
    print("4. Quick compare (text73 vs text1 and text2)")
    
    choice = input("\nEnter your choice (1-4): ").strip()
    
    if choice == "1":
        file1 = input("Enter path to first file: ").strip()
        file2 = input("Enter path to second file: ").strip()
        
        text1 = read_file_content(file1)
        text2 = read_file_content(file2)
        
        compare_texts(text1, text2, file1, file2)
    
    elif choice == "2":
        file_path = input("Enter path to file: ").strip()
        max_lines = input("Max lines to show (default 50): ").strip()
        max_lines = int(max_lines) if max_lines.isdigit() else 50
        
        show_file_content(file_path, max_lines)
    
    elif choice == "3":
        directory = input("Enter directory path: ").strip()
        pattern = input("File pattern (default *.txt): ").strip() or "*.txt"
        
        files = find_files_in_directory(directory, pattern)
        print(f"\nFiles found in '{directory}':")
        for file in files:
            print(f"  - {file}")
    
    elif choice == "4":
        # Quick comparison for your specific case
        print("🚀 Quick comparison mode")
        
        # You can modify these paths based on your actual file locations
        files_to_compare = {
            "text73": input("Enter path to text73 file: ").strip(),
            "text1": input("Enter path to text1 file: ").strip(),
            "text2": input("Enter path to text2 file: ").strip()
        }
        
        texts = {}
        for name, path in files_to_compare.items():
            texts[name] = read_file_content(path)
        
        if all(text is not None for text in texts.values()):
            # Compare text73 vs text1
            compare_texts(texts["text73"], texts["text1"], "text73", "text1")
            
            # Compare text73 vs text2
            compare_texts(texts["text73"], texts["text2"], "text73", "text2")
            
            # Compare text1 vs text2
            compare_texts(texts["text1"], texts["text2"], "text1", "text2")
    
    else:
        print("Invalid choice. Please run the script again.")

if __name__ == "__main__":
    main()