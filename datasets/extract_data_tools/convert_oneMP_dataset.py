"""
Install conda environment with https://github.com/conda-forge/blosc-hdf5-plugin-feedstock

python -m datasets.extract_data_tools.convert_oneMP_dataset --source_path <path>
      --target_path <path>

"""
import os
import h5py
import tqdm
import shutil
import argparse
import numpy as np

from datasets.prophesee_utils import dat_events_tools


def extract_events(path):
    file_handle = open(path, "rb")
    ev_start, ev_type, ev_size, img_size = dat_events_tools.parse_header(file_handle)
    file_handle.seek(ev_start)

    dat_event = np.fromfile(file_handle, dtype=[('ts', 'u4'), ('_', 'i4')])
    file_handle.close()

    x = np.bitwise_and(dat_event["_"], 16383)
    y = np.right_shift(np.bitwise_and(dat_event["_"], 268419072), 14)
    p = np.right_shift(np.bitwise_and(dat_event["_"], 268435456), 28)
    p = 2 * p - 1
    events_dict = {}
    events_dict['p'] = p
    events_dict['t'] = dat_event['ts']
    events_dict['x'] = x
    events_dict['y'] = y

    return events_dict


def save_to_h5(target_path, data: dict):
    assert not os.path.exists(target_path)
    filter_id = 32001  # Blosc

    compression_level = 1 # {0, ..., 9}
    shuffle = 2 # {0: none, 1: byte, 2: bit}
    # From https://github.com/Blosc/c-blosc/blob/7435f28dd08606bd51ab42b49b0e654547becac4/blosc/blosc.h#L66-L71
    # define BLOSC_BLOSCLZ   0
    # define BLOSC_LZ4       1
    # define BLOSC_LZ4HC     2
    # define BLOSC_SNAPPY    3
    # define BLOSC_ZLIB      4
    # define BLOSC_ZSTD      5
    compressor_type = 5
    compression_opts=(0, 0, 0, 0, compression_level, shuffle, compressor_type)

    p = data['p']
    t = data['t']
    x = data['x']
    y = data['y']
    # t_offset = data['t_offset']
    # ms_to_idx = data['ms_to_idx']

    with h5py.File(str(target_path), 'w') as h5f:
        ev_group = 'events'
        h5f.create_dataset('{}/p'.format(ev_group), data=p, compression=filter_id, compression_opts=compression_opts, chunks=True)
        h5f.create_dataset('{}/t'.format(ev_group), data=t, compression=filter_id, compression_opts=compression_opts, chunks=True)
        h5f.create_dataset('{}/x'.format(ev_group), data=x, compression=filter_id, compression_opts=compression_opts, chunks=True)
        h5f.create_dataset('{}/y'.format(ev_group), data=y, compression=filter_id, compression_opts=compression_opts, chunks=True)


def main():
    parser = argparse.ArgumentParser(description='Train network.')
    parser.add_argument('--source_path', help='Path to the directory containing .dat files', required=True)
    parser.add_argument('--target_path', help='Path to the directory containing .dat files', required=True)

    args = parser.parse_args()
    source_path = args.source_path
    target_path = args.target_path

    for file_name in tqdm.tqdm(os.listdir(source_path)):
        if file_name[-4:] == '.dat':
            events_dict = extract_events(os.path.join(source_path, file_name))
            save_to_h5(os.path.join(target_path, file_name[:-4] + '.h5'), events_dict)
        elif file_name[-4:] == '.npy':
            shutil.copyfile(os.path.join(source_path, file_name), os.path.join(target_path, file_name))


if __name__ == "__main__":
    main()
