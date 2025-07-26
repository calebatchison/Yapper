from dp.phonemizer import Phonemizer

from functools import partial
import torch
torch.load = partial(torch.load, weights_only=False)

phonemizer = Phonemizer.from_checkpoint('en_us_cmudict_ipa_forward.pt')
print(phonemizer('Its impossible to phonemize english!', lang='en_us'))