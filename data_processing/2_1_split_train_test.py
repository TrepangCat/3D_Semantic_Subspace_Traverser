import os
from tqdm import tqdm

if __name__ == '__main__':
    pathes = [
        '/home/brl/data_disk_4t/data/ShapeNetCore.v1/02691156',
        '/home/brl/data_disk_4t/data/ShapeNetCore.v1/02958343',
        '/home/brl/data_disk_4t/data/ShapeNetCore.v1/03001627',
        '/home/brl/data_disk_4t/data/ShapeNetCore.v1/04090263',
        '/home/brl/data_disk_4t/data/ShapeNetCore.v1/04379243',
    ]
    split = 0.8

    for path in tqdm(pathes):
        name_list = os.listdir(path)
        name_list = sorted(name_list)  # 排序

        cat_name = os.path.basename(path)

        split_point = int(len(name_list) * split)

        train_list = name_list[0:split_point]
        with open(path + '_train.txt', 'w') as f:
            for name in train_list:
                f.writelines(f'{name}\n')

        test_list = name_list[split_point:]
        with open(path + '_test.txt', 'w') as f:
            for name in test_list:
                f.writelines(f'{name}\n')

    print('finished')
