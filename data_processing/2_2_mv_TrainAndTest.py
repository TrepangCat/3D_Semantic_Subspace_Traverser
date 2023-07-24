import os
import glob
from tqdm import tqdm




if __name__ == '__main__':
    cat_list = [
        # '02691156',
        # '02958343',
        # '03001627',
        '04090263',
        # '04379243',
    ]

    split_file_path = '/home/brl/data_disk_4t/data/ShapeNetCore.v1'
    # data_basedir = '/home/brl/dataDisk2/wrw/dataset/ShapeNetDataSet_subset/ShapeNetCore.v1_processed_tmp'
    data_basedir = '/home/brl/dataDisk2/wrw/dataset/ShapeNetDataSet_subset/ShapeNetCore.v1_processed_largeScale'

    for cat in tqdm(cat_list, position=0):
        data_path = os.path.join(data_basedir, cat)

        # split train data
        train_list = []
        with open(os.path.join(split_file_path, cat + '_train.txt'), 'r') as f:
            for line in f:
                name = line.strip()
                train_list.append(name)

        train_dir = os.path.join(data_path, 'train')
        if not os.path.exists(train_dir):
            os.makedirs(train_dir)

        for name in tqdm(train_list, position=0):
            name_path = os.path.join(data_path, name)
            dirs = glob.glob(os.path.join(name_path + '*'), recursive=True)
            for dir in dirs:
                os.system(f'mv {dir} {train_dir}')

        # split test data
        test_list = []
        with open(os.path.join(split_file_path, cat + '_test.txt'), 'r') as f:
            for line in f:
                name = line.strip()
                test_list.append(name)

        test_dir = os.path.join(data_path, 'test')
        if not os.path.exists(test_dir):
            os.makedirs(test_dir)

        for name in tqdm(test_list, position=0):
            name_path = os.path.join(data_path, name)
            dirs = glob.glob(os.path.join(name_path + '*'), recursive=True)
            for dir in dirs:
                os.system(f'mv {dir} {test_dir}')
