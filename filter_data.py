from utils import process_dataset, process_image
import h5py

def main():
    train_data, train_label = process_dataset.extract_hdf5("/home/adrianwong/Projects/ML_localdata/Dataset/HDF5/CASIA_SC_L_A_label_00000_03880.hdf5")
    train_indices_partial = train_label < 1000
    train_data = train_data[train_indices_partial]
    train_label = train_label[train_indices_partial]

    print(train_data.shape)
    print(max(train_label))

    result_h5 = h5py.File("outputs/casia1000_train.hdf5", 'w')
    result_h5.create_dataset("data", data=train_data)
    result_h5.create_dataset("label", data=train_label)
    result_h5.close()
    print("done train")

    train_data = None
    train_label = None

    test_data, test_label = process_dataset.extract_hdf5("/home/adrianwong/Projects/ML_localdata/Dataset/HDF5/CASIA_SC_C_0.hdf5")
    test_indices_partial = test_label < 1000
    test_data = test_data[test_indices_partial]
    test_label = test_label[test_indices_partial]
    print(test_data.shape)
    print(max(test_label))

    result_h5 = h5py.File("outputs/casia1000_test.hdf5", 'w')
    result_h5.create_dataset("data", data=test_data)
    result_h5.create_dataset("label", data=test_label)
    result_h5.close()
    print("done test")

main()