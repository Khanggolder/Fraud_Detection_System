from collections import Counter
import re

def get_word_frequency(text: str) -> dict:
    """
    Calculates the frequency of each word in the provided text.
    
    Args:
        text (str): The input string to analyze.
        
    Returns:
        dict: A dictionary mapping words to their respective counts.
    """
    # Normalize text: lowercasing and removing non-alphanumeric characters
    clean_text = re.sub(r'[^\w\s]', '', text.lower())
    words = clean_text.split()
    
    return dict(Counter(words))

if __name__ == "__main__":
    sample_text = "apple orange apple banana orange apple"
    print(get_word_frequency(sample_text))