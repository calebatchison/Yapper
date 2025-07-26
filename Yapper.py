from dp.phonemizer import Phonemizer
from functools import partial
import torch

class Yapper:
    
    def speak(text):
        normalizedText = normalize(text)                # normalize
        phoneticText = phoneticize(normalizedText)      # Grapheme to Phoneme
        # Prosody Modeling
        # Waveform Synthesis
        return ""
        
    def normalize(text):
        return ""
    
    def phoneticize(text):
        torch.load = partial(torch.load, weights_only=False)
        phonemizer = Phonemizer.from_checkpoint('en_us_cmudict_ipa_forward.pt')
        return phonemizer(text, lang='en_us')