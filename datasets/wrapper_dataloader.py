from torch.utils.data import Dataset


class WrapperDataset(Dataset):
    def __init__(self, dataloader_a, dataloader_b, device, dataset_len_to_use=None):
        self.dataloader_a = dataloader_a
        self.dataloader_b = dataloader_b
        self.dataloader_a_iter = iter(self.dataloader_a)
        self.dataloader_b_iter = iter(self.dataloader_b)
        self.device = device

        self.dataset_a_larger = False
        if self.dataloader_a.__len__() > self.dataloader_b.__len__():
            self.dataset_a_larger = True

        if dataset_len_to_use == 'first':
            self.dataset_a_larger = True
        elif dataset_len_to_use == 'second':
            self.dataset_a_larger = False

    def __len__(self):
        if self.dataset_a_larger:
            return self.dataloader_a.__len__()
        else:
            return self.dataloader_b.__len__()

    def createIterators(self):
        self.dataloader_a_iter = iter(self.dataloader_a)
        self.dataloader_b_iter = iter(self.dataloader_b)

    def __getitem__(self, idx):
        """
        Returns two samples
        """
        if self.dataset_a_larger:
            try:
                dataset_b_data, dataset_b_label = next(self.dataloader_b_iter)
            except StopIteration:
                self.dataloader_b_iter = iter(self.dataloader_b)
                dataset_b_data, dataset_b_label = next(self.dataloader_b_iter)
            dataset_a_data, dataset_a_label = next(self.dataloader_a_iter)
        else:
            try:
                dataset_a_data, dataset_a_label = next(self.dataloader_a_iter)
            except StopIteration:
                self.dataloader_a_iter = iter(self.dataloader_a)
                dataset_a_data, dataset_a_label = next(self.dataloader_a_iter)
            dataset_b_data, dataset_b_label = next(self.dataloader_b_iter)

        return [dataset_a_data.to(self.device), dataset_a_label.to(self.device)], \
               [dataset_b_data.to(self.device), dataset_b_label.to(self.device)]
