#detectors.py
import hashlib

class PlagiarismDetector:
    def __init__(self, k_gram_len=5, window_size=4):
        self.k = k_gram_len
        self.w = window_size

    def _generate_k_grams(self, text):
        text = "".join(text.split())
        if len(text) < self.k:
            return [text]
        return [text[i:i+self.k] for i in range(len(text) - self.k + 1)]

    def _hash_k_grams(self, k_grams):
        hashes = []
        for kg in k_grams:
            hash_val = int(hashlib.md5(kg.encode('utf-8')).hexdigest(), 16)
            hashes.append(hash_val)
        return hashes

    def _winnowing(self, hashes):
        fingerprints = []
        if len(hashes) < self.w:
            fingerprints.append((min(hashes), 0))
            return fingerprints
            
        for i in range(len(hashes) - self.w + 1):
            window = hashes[i:i+self.w]
            min_val = min(window)
            fingerprints.append(min_val)
        return set(fingerprints)

    def get_fingerprint(self, code):
        k_grams = self._generate_k_grams(code)
        hashes = self._hash_k_grams(k_grams)
        return self._winnowing(hashes)

    def calculate_similarity(self, code1, code2):
        fp1 = self.get_fingerprint(code1)
        fp2 = self.get_fingerprint(code2)
        
        if not fp1 or not fp2:
            return 0.0
            
        intersection = len(fp1.intersection(fp2))
        union = len(fp1.union(fp2))
        
        return intersection / union if union > 0 else 0.0
