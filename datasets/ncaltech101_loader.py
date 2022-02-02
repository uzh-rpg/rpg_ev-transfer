import os
import torch
import random
import numpy as np

import datasets.data_util as data_util
from datasets.caltech101_loader import Caltech101RGB


class NCaltech101Events(Caltech101RGB):
    def getRootPath(self, root):
        """Function makes it easier to handle child of this class e.g. N-Caltech101 dataloader"""
        self.extended_data = False  # There is no extended data for events
        return os.path.join(root, 'N-Caltech101')

    def __getitem__(self, idx):
        """
        returns events and label, loading events from aedat
        :param idx:
        :return: x,y,t,p,  label
        """
        label = self.labels[idx]
        filename = self.files[idx]
        events = np.load(os.path.join(self.root, filename)).astype(np.float32)
        # Convert negative polarity to -1
        events[:, -1] = 2 * events[:, -1] - 1
        nr_events = events.shape[0]

        window_start = 0

        if self.augmentation:
            events = data_util.random_shift_events(events)
            events = data_util.random_flip_events_along_x(events)
            nr_events = events.shape[0]
            window_start = random.randrange(0, max(1, nr_events - self.nr_events_window))

        if self.nr_events_window != -1:
            # Catch case if number of events in batch is lower than number of events in window.
            window_end = min(nr_events, window_start + self.nr_events_window)
        else:
            window_start = 0
            window_end = nr_events

        events = events[window_start:window_end, :]

        event_tensor = data_util.generate_input_representation(events, self.event_representation,
                                                               (self.height, self.width),
                                                               nr_temporal_bins=self.nr_temporal_bins)
        event_tensor = torch.from_numpy(data_util.normalize_event_tensor(event_tensor))

        if self.augmentation:
            mid_point = np.max(events[:, :2], axis=0) // 2
            event_tensor = data_util.random_crop_resize(event_tensor, mid_point)

        return event_tensor, label
