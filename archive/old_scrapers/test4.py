import string

def word_count(strings):

    counts = {}
    
    for s in strings:
        s_clean = s.translate(str.maketrans('', '', string.punctuation))
        words = s_clean.lower().split()
        for word in words:
            counts[word] = counts.get(word, 0) + 1
            
    return counts
