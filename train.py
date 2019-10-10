import numpy as np
from pathlib import Path
from keras.callbacks import ModelCheckpoint, CSVLogger, EarlyStopping, ReduceLROnPlateau
from utils import nii_utils, yaml_utils
from model import unet_3d
from random import shuffle
from skimage import transform

output_path = Path('E:/Dataset/BraTS_2018')


def create_data_yaml(path):
    if Path(str(output_path / 't22seg_train.yaml')).exists():
        return
    paired_data = list()
    path = Path(path) / 't2'
    for t1_file in path.iterdir():
        seg_file = str(t1_file).replace('t2', 'seg')
        t1_image = nii_utils.nii_reader(str(t1_file))
        seg_image = nii_utils.nii_reader(str(seg_file))
        if t1_image.shape == seg_image.shape:  # check dataset
            paired_data.append({'t2': str(t1_file), 'seg': str(seg_file)})
    shuffle(paired_data)
    yaml_utils.write(str(output_path / 't22seg_train.yaml'), paired_data[:8 * len(paired_data) // 10])  # train 80%
    yaml_utils.write(str(output_path / 't22seg_test.yaml'), paired_data[8 * len(paired_data) // 10:])  # test 20%


def data_generator(data_list, batch_size):
    batch_x_list = list()
    batch_y_list = list()
    while True:
        for i in data_list:
            t2_model = nii_utils.nii_reader(i['t2'])
            t2_model = transform.resize(t2_model, (64, 64, 32))
            seg_model = nii_utils.nii_reader(i['seg'])
            seg_model = transform.resize(seg_model, (64, 64, 32))
            batch_x_list.append([t2_model])
            batch_y_list.append([seg_model])
            if len(batch_x_list) == batch_size:
                yield np.asarray(batch_x_list), np.asarray(batch_y_list)
                batch_x_list = list()
                batch_y_list = list()


def data_loader():
    train_list = yaml_utils.read(str(output_path / 't22seg_train.yaml'))
    train_generator = data_generator(train_list, batch_size=6)

    test_list = yaml_utils.read(str(output_path / 't22seg_test.yaml'))
    test_generator = data_generator(test_list, batch_size=12)
    return train_generator, len(train_list), test_generator, len(test_list)


if __name__ == '__main__':
    create_data_yaml(output_path)  # first deal with dataset

    train_generator, train_steps, validation_generator, validation_steps = data_loader()  # second create generator

    _model = unet_3d(input_shape=(1, 64, 64, 32))  # third create model (channels,x,y,z)

    Path('_').mkdir(parents=True, exist_ok=True)  # create file in fold _ for finding and deleting  easily
    _model.fit_generator(generator=train_generator, steps_per_epoch=train_steps, epochs=200,  # final train model
                         validation_data=validation_generator, validation_steps=validation_steps,
                         callbacks=[ModelCheckpoint('_/tumor_segmentation_model.h5', save_best_only=True),
                                    CSVLogger('_/training.log', append=True),
                                    ReduceLROnPlateau(factor=0.5, patience=50, verbose=1),
                                    EarlyStopping(verbose=1, patience=None)])
