
# Add this to your pipeline.py to escape problematic regex patterns:

def escape_regex_patterns(text):
    """Escape patterns that cause 'invalid group reference' errors"""
    if not isinstance(text, str):
        return str(text)
    
    # Escape dollar followed by digits (group references)
    text = re.sub(r'\$([0-9]+)', r'\\$\1', text)
    
    # Escape backslash followed by digits (backreferences)  
    text = re.sub(r'\\([0-9]+)', r'\\\\\1', text)
    
    return text

# Use this at the start of process_text():
def process_text(self, text):
    text = escape_regex_patterns(text)  # Add this line
    # ... rest of your existing code
        