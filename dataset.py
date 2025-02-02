# Author: Firat Ozdemir, May 2022, firat.ozdemir@datascience.ch
import numpy as np
import h5py

class Dataset_Paired_Input_Output:
    '''Samples (in_key, out_key) sample pairs from fname_h5 and applies transforms on in_key and transforms_target on out_key'''
    def __init__(self, fname_h5, in_key, out_key, inds=None, transforms=None, transforms_target=None, **kwargs):
        self.fname_h5 = fname_h5
        self.inds = inds
        self.in_key = in_key
        self.out_key = out_key
        self.transforms = transforms
        self.transforms_target = transforms_target
        self.prng = kwargs.get('prng', np.random.RandomState(42))
        self.shuffle = kwargs.get('shuffle', False)
        self.len = None #will be overwritten in check_data()
        self.check_data()

    def check_data(self,):
        len_ = None
        with h5py.File(self.fname_h5, 'r') as fh:
            for k in [self.in_key, self.out_key]:
                if len_ is None: 
                    len_ = fh[k].shape[0]
                if len_ != fh[k].shape[0]:
                    raise AssertionError('Length of datasets vary across keys. %d vs %d' % (len_, fh[k].shape[0]))
        if self.inds is None:
            self.len = len_
            self.inds = np.arange(len_)
        else:
            self.len = len(self.inds)

    def __len__(self):
        return self.len

    def __getitem__(self, index): 
        with h5py.File(self.fname_h5, 'r') as fh:
            x = fh[self.in_key][index,...]
            if self.transforms is not None:
                x = self.transforms(x)

            y = fh[self.out_key][index,...]
            if self.transforms_target is not None:
                y = self.transforms_target(y)
        return (x,y)

    def __iter__(self):
        inds = np.copy(self.inds)
        if self.shuffle:
            self.prng.shuffle(inds)
        for i in inds:
            s = self.__getitem__(index=i)
            yield s

class Dataset:
    def __init__(self, fname_h5, key, transforms, inds, label=None, shuffle=False, **kwargs):
        self.fname_h5 = fname_h5
        self.key = key
        self.inds = inds
        self.transforms = transforms
        self.shuffle = shuffle
        self.prng = kwargs.get('prng', np.random.RandomState(42))
        self.len = None
        self._check_data()
        self.label = label
    
    def _check_data(self,):
        len_ = None
        l_keys = self.key if isinstance(self.key, list) else [self.key]
        with h5py.File(self.fname_h5, 'r') as fh:
            for k in l_keys:
                if len_ is None: 
                    len_ = fh[k].shape[0]
                if len_ != fh[k].shape[0]:
                    raise AssertionError('Length of datasets vary across keys. %d vs %d' % (len_, fh[k].shape[0]))
        if self.inds is None:
            self.len = len_
            self.inds = np.arange(len_)
        else:
            self.len = len(self.inds)

    def __len__(self):
        return self.len

    def __getitem__(self, index): 
        with h5py.File(self.fname_h5, 'r') as fh:
            # x = fh[self.key][index,...]
            x = fh[self.key][self.inds[index], ...]  # Use self.inds to map indices
            x = x[None,...] ## add a channel dimension [1, H, W]
            if self.transforms is not None:
                x = self.transforms(x)
        if self.label is not None:
            return x, self.label
        return x

    ## This is not working if using torch.utils.data.DataLoader since the dataloader use __getitem__ to fetch data:
    # def __iter__(self):
    #     print("Using __iter__ of the customized dataset!")
    #     inds = np.copy(self.inds)
    #     if self.shuffle:
    #         self.prng.shuffle(inds)
    #     for i in inds:
    #         s = self.__getitem__(index=i)
    #         yield s