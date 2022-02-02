import torch
import numpy as np

from torch.utils.data.dataloader import default_collate


class ObjectDetLoader:
    def __init__(self, dataset, batch_size, num_workers, pin_memory, device=None, shuffle=True, drop_last=True):
        self.device = device
        split_indices = list(range(len(dataset)))
        if shuffle:
            sampler = torch.utils.data.sampler.SubsetRandomSampler(split_indices)
            self.loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=sampler,
                                                      num_workers=num_workers, pin_memory=pin_memory,
                                                      collate_fn=self.collate_bboxes, drop_last=drop_last)
        else:
            self.loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                      num_workers=num_workers, pin_memory=pin_memory,
                                                      collate_fn=self.collate_bboxes, drop_last=drop_last)


    def __iter__(self):
        for data in self.loader:
            yield data

    def __len__(self):
        return len(self.loader)

    def collate_bboxes(self, data):
        bboxes = []
        event_tensor = []
        for i, d in enumerate(data):
            event_tensor.append(d[0])
            bbox = np.concatenate([i*np.ones((len(d[1]), 1), dtype=np.int32), d[1]], 1)
            bboxes.append(bbox)

        bboxes = torch.from_numpy(np.concatenate(bboxes, 0))
        event_tensor = default_collate(event_tensor)

        return event_tensor, bboxes
