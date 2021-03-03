import torch
import os
import hparam


root_path = '/Users/karanbhatia/Desktop/karan/Samsung PRISM/vq-vae/ZeroSpeech/output_data/'


class Dataset(torch.utils.data.Dataset):
    def __init__(root_path):
        super(Dataset, self).__init__()

        wav = np.load(root_path.with_suffix('.wav'))
        mel = np.load(root_path.with_suffix('.mel'))
