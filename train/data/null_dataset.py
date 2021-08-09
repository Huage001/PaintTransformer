from data.base_dataset import BaseDataset
import os


class NullDataset(BaseDataset):

    def __init__(self, opt):
        BaseDataset.__init__(self, opt)

    def __getitem__(self, index):
        return {'A_paths': os.path.join(self.opt.dataroot, '%d.jpg' % index)}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return self.opt.max_dataset_size
