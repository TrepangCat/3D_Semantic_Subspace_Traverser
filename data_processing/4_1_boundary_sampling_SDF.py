#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

import argparse
import json
import logging
import os
from glob import glob
from multiprocessing import Pool

import numpy as np
from tqdm import tqdm

import deep_sdf
import deep_sdf.workspace as ws


def filter_classes_glob(patterns, classes):
    import fnmatch

    passed_classes = set()
    for pattern in patterns:
        passed_classes = passed_classes.union(
            set(filter(lambda x: fnmatch.fnmatch(x, pattern), classes))
        )

    return list(passed_classes)


def filter_classes_regex(patterns, classes):
    import re

    passed_classes = set()
    for pattern in patterns:
        regex = re.compile(pattern)
        passed_classes = passed_classes.union(set(filter(regex.match, classes)))

    return list(passed_classes)


def filter_classes(patterns, classes):
    if patterns[0] == "glob":
        return filter_classes_glob(patterns, classes[1:])
    elif patterns[0] == "regex":
        return filter_classes_regex(patterns, classes[1:])
    else:
        return filter_classes_glob(patterns, classes)


def process_mesh_for_pool(input_dict):
    source_obj = input_dict['source_obj']
    out_npz = input_dict['out_npz']
    executable = input_dict['executable']
    specific_args = input_dict['specific_args']
    num_points = input_dict['num_points']

    process_mesh(source_obj, out_npz, executable, specific_args, num_points)

    return True


def process_mesh(mesh_filepath, target_filepath, executable, additional_args, num_points):
    logging.info(mesh_filepath + " --> " + target_filepath)
    command_list = [executable, "-m", mesh_filepath, "-o", target_filepath] + additional_args

    command = ""
    for s in command_list:
        command = command + s + ' '

    os.system(command)

    assert os.path.exists(target_filepath)

    samples = np.load(target_filepath)
    os.system(f'rm {target_filepath}')
    p = samples['pos']
    n = samples['neg']

    # rescale the points
    # 为了匹配 pc_norm.obj 的尺寸（点的最大值0.5，最小值-0.5），对点的值域做处理
    scale = abs(n[:, 0:3]).max()
    p[:, 0:3] = (p[:, 0:3] / scale) * 0.999
    n[:, 0:3] = (n[:, 0:3] / scale) * 0.999

    num_samples = int(num_points / 2)

    num = min(len(p), num_samples)
    order = np.random.choice(len(p), num, replace=False)
    p = p[order]

    num = min(len(n), num_samples)
    order = np.random.choice(len(n), num, replace=False)
    n = n[order]

    target_filepath_final = target_filepath[0:-4] + f'{num_points}.npz'
    np.savez(target_filepath_final, pos=p, neg=n)

    return True



def append_data_source_map(data_dir, name, source):
    data_source_map_filename = ws.get_data_source_map_filename(data_dir)

    print("data sources stored to " + data_source_map_filename)

    data_source_map = {}

    if os.path.isfile(data_source_map_filename):
        with open(data_source_map_filename, "r") as f:
            data_source_map = json.load(f)

    if name in data_source_map:
        if not data_source_map[name] == os.path.abspath(source):
            raise RuntimeError(
                "Cannot add data with the same name and a different source."
            )

    else:
        data_source_map[name] = os.path.abspath(source)

        with open(data_source_map_filename, "w") as f:
            json.dump(data_source_map, f, indent=2)


if __name__ == "__main__":

    os.environ['MESA_GL_VERSION_OVERRIDE'] = '3.3'

    arg_parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="Pre-processes data from a data source and append the results to "
                    + "a dataset.",
    )
    arg_parser.add_argument(
        "--source",
        "-s",
        dest="source_dir",
        # required=True,
        # default='/home/brl/dataDisk2/wrw/dataset/ShapeNetDataSet_subset/ShapeNetCore.v1_processed/03001627/train/origin',
        default='/home/brl/dataDisk2/wrw/dataset/ShapeNetDataSet_subset/ShapeNetCore.v1_processed/03001627/train/extend',
        help="The directory which holds the data to preprocess and append.",
    )
    arg_parser.add_argument(
        "--skip",
        dest="skip",
        default=False,
        action="store_true",
        help="If set, previously-processed shapes will be skipped",
    )
    arg_parser.add_argument(
        "--threads",
        dest="num_threads",
        default=8,
        help="The number of threads to use to process the data.",
    )
    arg_parser.add_argument(
        "--test",
        "-t",
        dest="test_sampling",
        default=False,
        action="store_true",
        help="If set, the script will produce SDF samplies for testing",
    )
    arg_parser.add_argument(
        "--surface",
        dest="surface_sampling",
        default=False,
        action="store_true",
        help="If set, the script will produce mesh surface samples for evaluation. "
             + "Otherwise, the script will produce SDF samples for training.",
    )
    arg_parser.add_argument(
        "--num_points",
        default=200000,
        help="num of the samples points",
    )

    deep_sdf.add_common_args(arg_parser)

    args = arg_parser.parse_args()

    deep_sdf.configure_logging(args)

    additional_general_args = []

    # 设置可执行程序
    deepsdf_dir = os.path.dirname(os.path.abspath(__file__))
    if args.surface_sampling:
        executable = os.path.join(deepsdf_dir, "bin/SampleVisibleMeshSurface")
        subdir = ws.surface_samples_subdir
        extension = ".ply"
    else:
        executable = os.path.join(deepsdf_dir, "bin/PreprocessMesh")
        subdir = ws.sdf_samples_subdir
        extension = ".npz"

        if args.test_sampling:
            additional_general_args += ["-t"]

    logging.info(
        "Preprocessing data from "
        + args.source_dir
    )

    if args.surface_sampling:
        # 这里面应该是没有做修改的
        normalization_param_dir = os.path.join(
            args.data_dir, ws.normalization_param_subdir, args.source_name
        )
        if not os.path.isdir(normalization_param_dir):
            os.makedirs(normalization_param_dir)

    # append_data_source_map(args.data_dir, args.source_name, args.source_dir)
    # class_directories = split[args.source_name]

    meshes_targets_and_specific_args = []
    instance_dirs = args.source_dir

    obj_list = glob(args.source_dir + '/**/*pc_norm.obj', recursive=True)

    for obj_path in obj_list:

        processed_filepath = os.path.join(os.path.dirname(obj_path),
                                          f'boundary_SDF_samples{args.num_points}' + extension)
        if args.skip and os.path.isfile(processed_filepath):
            logging.debug("skipping " + processed_filepath)
            continue

        processed_filepath_tmp = os.path.join(os.path.dirname(obj_path), f'boundary_SDF_samples' + extension)

        try:
            specific_args = []

            if args.surface_sampling:
                # 这里面应该是没有做修改的，这里得查
                # normalization_param_target_dir = os.path.join(
                #     normalization_param_dir, class_dir
                # )
                #
                # if not os.path.isdir(normalization_param_target_dir):
                #     os.mkdir(normalization_param_target_dir)
                #
                # normalization_param_filename = os.path.join(
                #     normalization_param_target_dir, instance_dir + ".npz"
                # )
                # specific_args = ["-n", normalization_param_filename]

                print('该部分未作修改，请查阅 DeepSDF 代码，做相关修改')
                exit()

            meshes_targets_and_specific_args.append(
                {
                    'source_obj': obj_path,
                    'out_npz': processed_filepath_tmp,
                    'specific_args': specific_args + additional_general_args,
                    'num_points': args.num_points,
                    'executable': executable,
                }
            )

        # 下面这个目前对于我的代码来说，已经没有用了
        except deep_sdf.data.NoMeshFileError:
            logging.warning("No mesh found for instance " + obj_path)
        except deep_sdf.data.MultipleMeshFileError:
            logging.warning("Multiple meshes found for instance " + obj_path)

    # with concurrent.futures.ThreadPoolExecutor(
    #     max_workers=int(args.num_threads)
    # ) as executor:
    #
    #     for (
    #         mesh_filepath,
    #         target_filepath,
    #         specific_args,
    #         num_points,
    #     ) in meshes_targets_and_specific_args:
    #         executor.submit(
    #             process_mesh,
    #             mesh_filepath,
    #             target_filepath,
    #             executable,
    #             specific_args + additional_general_args,
    #             num_points
    #         )
    #
    #     executor.shutdown()

    # with Pool(int(args.num_threads)) as p:
    with Pool(12) as p:
        res_list = p.imap_unordered(process_mesh_for_pool, meshes_targets_and_specific_args, chunksize=2)

        bar = tqdm(range(len(meshes_targets_and_specific_args)))
        for res in res_list:
            bar.update()

    print("finished")
