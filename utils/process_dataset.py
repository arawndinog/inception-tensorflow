import h5py
import numpy as np

def extract_hdf5(file_path, bchw2bhwc=True):
    with h5py.File(file_path,'r') as hdf5_data:
        data = np.array(hdf5_data.get('data'))
        label = np.array(hdf5_data.get('label'),dtype=np.int32).squeeze()
        if bchw2bhwc:
            data = np.reshape(data,(data.shape[0],data.shape[2],data.shape[3],data.shape[1]))
        return data, label