import glob
import multiprocessing as mp
import os
import subprocess
import sys
from multiprocessing import Pool
import argparse
import trimesh
from tqdm import tqdm


# INPUT_PATH = 'shapenet/data'


def run_os_cmd(cmd):
    process = subprocess.Popen(
        [cmd],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        shell=True,
    )

    output, err = process.communicate()
    if len(err) > 0:
        print(err)
        sys.exit(1)

    return


def to_off(path):
    if os.path.exists(path + '/isosurf.off'):
        return

    if os.path.exists(path + '/isosurf_scaled.off'):
        return

    input_file = path + '/isosurf.obj'
    output_file = path + '/isosurf.off'

    cmd = 'meshlabserver -i {} -o {}'.format(input_file, output_file)
    # if you run this script on a server: comment out above line and uncomment the next line
    # cmd = 'xvfb-run -a -s "-screen 0 800x600x24" meshlabserver -i {} -o {}'.format(input_file,output_file)
    # run_os_cmd(cmd)
    os.system(cmd)


def scale(path):
    if os.path.exists(path + '/isosurf_scaled.off'):
        return

    try:
        mesh = trimesh.load(path + '/isosurf.off', process=False)
        total_size = (mesh.bounds[1] - mesh.bounds[0]).max()
        centers = (mesh.bounds[1] + mesh.bounds[0]) / 2

        mesh.apply_translation(-centers)
        mesh.apply_scale(1 / total_size)
        mesh.export(path + '/isosurf_scaled.off')

        os.system(f'rm {path}/isosurf.off')
    except:
        print('Error with {}'.format(path))

    # print('Finished {}'.format(path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--processed_data_dir', type=str, default='../sample_data/ShapeNetCore.v1_processed_tmp')
    FLAGS = parser.parse_args()

    input_path = FLAGS.processed_data_dir

    iso_dir_list = []
    iso_dir_list += glob.glob(os.path.join(input_path, '**', 'isosurf.obj'), recursive=True)
    iso_dir_list = [os.path.dirname(x) for x in iso_dir_list]


    def get_mission_from_list(list):
        for x in list:
            yield x


    with Pool(int(mp.cpu_count() / 2)) as p:
        # convert isosurf.obj to isosurf.off

        res_list = p.imap_unordered(to_off, get_mission_from_list(iso_dir_list), chunksize=4)
        bar = tqdm(range(len(iso_dir_list)), position=0)
        for res in res_list:
            bar.update()

        # normalize isosurf.off
        res_list = p.imap_unordered(scale, get_mission_from_list(iso_dir_list), chunksize=4)
        bar = tqdm(range(len(iso_dir_list)), position=0)
        for res in res_list:
            bar.update()
