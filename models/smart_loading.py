import torch
import numpy as np
import pandas as pd
import nibabel as nib
from sklearn.model_selection import StratifiedKFold
from torch.jit import Error

def genIdx():
    participants = pd.read_csv('../../ds000201-download/participants.tsv', sep='\t')
    exclude = ['sub-9016', 'sub-9022', 'sub-9044', 'sub-9066', 'sub-9078', 'sub-9095']
    rows = participants.iterrows()
    labels = []
    for idx, row in rows:
        pid = row['participant_id']
        if pid in exclude: continue
        labels.append((pid, 1))
        labels.append((pid, 2))

    np.array(labels, dtype=object)
    np.save('subjects.npy', labels)
    return

# NOTE: assumes dataset path and labels path
class SmartLoader():
    def __init__(self, nfolds, load_type=None, data_shape=None, device='cuda', dtype=torch.float32):
        # Data vars
        self.sub_list = np.load('subjects.npy') # (subject number, session number)
        self.label_list_cpu = torch.load('restLabels_meanTime_full_int8.pt',  map_location='cpu')
        self.label_list = self.label_list_cpu#torch.tensor(self.label_list_cpu, dtype=torch.long, device=device)
        self.data = None
        self.data_shape = data_shape
        # Handle data loading options
        self.load_type = load_type
        if load_type == None:
            self.preload = False
        elif type(load_type) == str:
            if load_type == 'none':
                self.preload = False
            elif load_type == 'single':
                self.preload = False
            elif load_type == 'batch':
                self.preload = True
            else:
                self.preload = True
        else:
            raise Error(f"'{load_type}' not recognized for keyword arg load_type. Please enter 'none', 'single', 'batch', or the path to the .pt file containing the dataset.")
        # CUDA vars
        self.device = device
        self.dtype = dtype
        # Training/Test vars
        self.skf = StratifiedKFold(n_splits=nfolds)
        self.training_sleep_accs = np.zeros(nfolds)
        self.training_rest_accs = np.zeros(nfolds)
        self.training_total_accs = np.zeros(nfolds)
        self.testing_sleep_accs = np.zeros(nfolds)
        self.testing_rest_accs = np.zeros(nfolds)
        self.testing_total_accs = np.zeros(nfolds)
        self.splits = self.skf.split(self.sub_list, self.label_list_cpu)
        self.splits_iter = iter(self.splits)
        self.train = None
        self.test = None
        self.train_len = 0
        self.test_len = 0
        return
    def __len__(self):
        return self.skf.get_n_splits()

    def __preprocess(self, data):
        if not self.preload:
            return torch.tensor(np.mean(data, axis=3), device=self.device, dtype=self.dtype).flatten()
        else:
            return torch.tensor(np.mean(data, axis=3), device='cpu', dtype=self.dtype).flatten()
    def __train_loop(self, f, model, epochs):
        model.train()
        if self.preload == True and self.load_type == 'batch':
            self.preload_train()
        running_loss = 0
        for e in range(epochs):
            for i in range(self.train_len):
                x = self.train_step(i, self.preload)#.float()
                yhat = model(x)
                #y = self.label_list[None, self.train[i]]
                y = torch.tensor([self.label_list[self.train[i]]], dtype=torch.long, device=self.device)
                running_loss += model.step(y, yhat)
            print(f'({f, e + 1}): Running loss: {running_loss / (self.train_len)}')
            running_loss = 0
        return
    def __train_eval_loop(self, f, model):
        # NOTE: Assumes self.data is training data
        model.eval()
        tp = 0 # correctly identifies sleep deprived
        tn = 0 # correctly identifies not sleep deprived
        fp = 0 # incorrectly identifies as sleep deprived
        fn = 0 # incorrectly identifies as not sleep deprived
        for i in range(self.train_len):
            x = self.train_step(i, self.preload)
            yhat = model(x)
            y = self.label_list[self.train[i]]
            if np.argmax(yhat.detach().cpu().numpy()) == y:
                if y == 0:
                    tp += 1
                else:
                    tn += 1
            else:
                if y == 1:
                    fp += 1
                else:
                    fn += 1
        acc = 100*((tp + tn) / (self.train_len))
        self.training_sleep_accs[f] = 100 * (tp / (tp + fp + 1e-9))
        self.training_rest_accs[f] =  100 * (tn / (tn + fn + 1e-9))
        self.training_total_accs[f] = acc
        print(f'({f}) Train Accuracy: {acc}')
        return
    def __test_eval_loop(self, f, model):
        model.eval()
        self.preload_test()
        tp = 0 # correctly identifies sleep deprived
        tn = 0 # correctly identifies not sleep deprived
        fp = 0 # incorrectly identifies as sleep deprived
        fn = 0 # incorrectly identifies as not sleep deprived
        for i in range(self.test_len):
            x = self.test_step(i, self.preload)
            yhat = model(x)
            y = self.label_list[self.test[i]]
            if np.argmax(yhat.detach().cpu().numpy()) == y:
                if y == 0:
                    tp += 1
                else:
                    tn += 1
            else:
                if y == 1:
                    fp += 1
                else:
                    fn += 1
        acc =  acc = 100*((tp + tn) / (self.test_len))
        self.testing_sleep_accs[f] = 100 * (tp / (tp + fn + 1e-9))
        self.testing_rest_accs[f] =  100 * (tn / (tn + fp + 1e-9))
        self.testing_total_accs[f] = acc
        print(f'({f}) Test Accuracy: {acc}')
        return

    def new_split(self, model):
        self.train, self.test = next(self.splits_iter)
        self.train_len = len(self.train)
        self.test_len = len(self.test)
        self.train_idx = 0
        for layer in model.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
        if self.preload:
            self.preload_train()
        return
    def preload_train(self):
        if self.load_type == 'batch':
            self.data = torch.empty(self.train_len, self.data_shape, device=self.device, dtype=self.dtype)
            for i in range(self.train_len):
                self.data[i] = self.train_step(i)
        else:
            self.data = torch.load(self.load_type, map_location=self.device).to(self.dtype)
        print("Finished loading training data.")
        return
    def preload_test(self):
        if self.load_type == 'batch':
            self.data = torch.empty(self.test_len, self.data_shape, device=self.device, dtype=self.dtype)
            for i in range(self.test_len):
                self.data[i] = self.test_step(i)
            print("Finished loading testing data.")
        return   
    def train_step(self, i, preloaded=False):
        if not preloaded:
            scanType = 'func/'
            task = '_task-rest_bold.nii.gz'
            sub = self.sub_list[self.train[i]][0]
            ses = self.sub_list[self.train[i]][1]
            path = '../../ds000201-download/' + sub + '/ses-' + ses + '/' + scanType + sub + '_ses-' + ses + task
            img = nib.load(path)
            data = np.asarray(img.dataobj)
            return self.__preprocess(data)
        else:
            return self.data[self.train[i]]
    def test_step(self, i, preloaded=False):
        if not preloaded:
            scanType = 'func/'
            task = '_task-rest_bold.nii.gz'
            sub = self.sub_list[self.test[i]][0]
            ses = self.sub_list[self.test[i]][1]
            path = '../../ds000201-download/' + sub + '/ses-' + ses + '/' + scanType + sub + '_ses-' + ses + task
            img = nib.load(path)
            data = np.asarray(img.dataobj)
            return self.__preprocess(data)
        else:
            return self.data[self.test[i]]
    def run(self, model, epochs):
        if self.dtype == torch.float16:
            model.half()
        for f in range(len(self)):
            print(f'Starting fold {f}')
            print("==========================")
            self.new_split(model)
            self.__train_loop(f, model, epochs)
            print(f'Fold {f} Results')
            print("--------------------------")
            self.__train_eval_loop(f, model)
            self.__test_eval_loop(f, model)
            print("==========================")   
        print(f'Average Training Accuracies')
        print(f'Sleep Deprived: {np.mean(self.training_sleep_accs)}%\tWell Rested: {np.mean(self.training_rest_accs)}%')
        print(f'Total: {np.mean(self.training_total_accs)}%')
        print('')
        print(f'Average Testing Accuracies')
        print(f'Sleep Deprived: {np.mean(self.testing_sleep_accs)}%\tWell Rested: {np.mean(self.testing_rest_accs)}%')
        print(f'Total: {np.mean(self.testing_total_accs)}%')
        return


if __name__ == '__main__':
    genIdx()