import math
from pathlib import Path

import torch
import jax
import numpy as np
import nibabel as nib
import pandas as pd
import tensorly as tl
from tensorly.decomposition import tucker
from tqdm.notebook import tqdm

class BrainDataset(torch.utils.data.Dataset):
    
    def __init__(self,
                 data_path: str='../../ds000201-download/',
                 task: str='_task-rest_bold.nii.gz',
                 tsv_path: str='../../ds000201-download/participants.tsv',
                 time_idx: int=190,
                 exclude: list=None,):
        
        self.__dict__.update(locals())
        if not exclude:
            self.exclude = ['sub-9016', 'sub-9022', 'sub-9044', 'sub-9066', 'sub-9078', 'sub-9095']
        
        self._setup_()
        self.participants = pd.read_csv(tsv_path, sep='\t')   
    
    def _setup_(self):
        # Setup Paths
        paths = sorted(Path(self.data_path).rglob(f'*{self.task}'))
        
        self.paths = []
        for idx, p in enumerate(paths):
            if any([e in p.name for e in self.exclude]):
                print(f'Skipping: {p.name}')
                continue
            self.paths.append(p)
    
        # Set Train Test Split
        

    def __load_fmri__(self, path) -> torch.Tensor:
        
        trueShape = (128, 128, 49)
        img = nib.load(path)
        
        if img.shape[:-1] != trueShape:
            print(f"Error: Shape is {img.shape} not {trueShape}")
            return None
        
        data = img.get_fdata()[:, :, :, :self.time_idx]
        data = torch.tensor(data, dtype=torch.float32)
        
        # Repeat the last entry of the time series until it matches the expect dimensions
        # Happy batching, happy life
        if data.shape[-1] < self.time_idx:
            missing_dim = self.time_idx - data.shape[-1]
            repeat = torch.tensor([data[:, :, :, -1].tolist() for _ in range(missing_dim)])
            repeat = torch.movedim(repeat,0, -1)
            data = torch.cat((data, repeat), axis=-1)
        return data
        
        
    def __len__(self):
        return(len(self.paths))
    
    
    def __getitem__(self, idx) -> (torch.Tensor, torch.Tensor):
        p = self.paths[idx]
        pid = p.name.split('_')[0]
        sleep_deprived_label = torch.tensor((self.participants[self.participants['participant_id'] == pid]['Sl_cond']  - 1).item(), dtype=torch.long)

        data = self.__load_fmri__(str(p))
        return (data, sleep_deprived_label)

    


dataset = BrainDataset()

rng = np.random.default_rng(seed=42)
indices = list(range(len(dataset)))
rng.shuffle(indices)
split = math.floor(len(dataset) *.8)
train_slice = indices[:split]
test_slice = indices[split:]

training_set = torch.utils.data.Subset(dataset, train_slice)
eval_set = torch.utils.data.Subset(dataset, test_slice)

# training_dl = torch.utils.data.DataLoader(training_set, batch_size=32, shuffle=True) 
training_dl = torch.utils.data.DataLoader(training_set, batch_size=8, shuffle=False) 
eval_dl = torch.utils.data.DataLoader(eval_set, batch_size=32, shuffle=True) 


dataset.__load_fmri__('../../ds000201-download/sub-9047/ses-1/func/sub-9047_ses-1_task-rest_bold.nii.gz').shape


# Process Tucker Tensor Decomposition 
full_dl = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False) 
tl.set_backend('pytorch') # Or 'mxnet', 'numpy', 'tensorflow', 'cupy' or 'jax'
full_labels = None
full_data = None
full_factos = None


rank = 10
# rank = [5, 5, 5, 190]


def get_tucker(x, rank=5):
    core, factors = tucker(tl.tensor(x), rank=rank)
    return core.flatten(), core.flatten()

full_factors = []
for data in tqdm(full_dl):
    x, labels = data
    bs = x.shape[0]
    ts = x.shape[-1]
    
    # assert len(rank) == x.ndim
    # decom_x = torch.stack([get_tucker(tensor).core.flatten() for d in range(2)])
    # decom_x.shape
    
    reduce_data, factors = get_tucker(x.squeeze(), rank)
    if full_labels is not None:
        full_labels = torch.vstack([full_labels, labels])
        full_data = torch.vstack([full_data, reduce_data])
        full_factors = torch.vstack([full_factors, factors])
    else:
        full_labels = labels
        full_data = reduce_data
        full_factors = factors
        
data_dict = {'data': full_data.view(160, -1),
 'factors': full_factors.view(160, -1),
 'labels': full_labels}

torch.save(data_dict, 'task_rest.pt')