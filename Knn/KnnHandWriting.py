from numpy import *


class c()
    def __init__(self, kval):



    def create_data_set(self, train_images_file, train_label_file, test_images_file, test_label_file):
        self.imgReader = imgReader.TrainDataSet(train_images_file, train_label_file, test_images_file,
                                                    test_label_file)
        self.imgReader.read_train_img()
        self.imgReader.read_train_label()
        self.imgReader.read_test_img()
        self.imgReader.read_test_label()
