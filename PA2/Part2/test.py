from data_loader import smile_dataset_clear, \
                            smile_dataset_blur, \
                            data_loader_mnist
import time
datasets = [(smile_dataset_clear(), 'Clear smile data', 3)
                ,(smile_dataset_blur(), 'Blur smile data', 3)
                ,(data_loader_mnist(), 'MNIST', 10)]

for data, name, num_classes in datasets:
    print('%s: %d class classification' % (name, num_classes))
    X_train, X_test, y_train, y_test = data
