""" Visualization of SensatUrban dataset
Author: Zhipeng Jiang
Date: September 2023
"""

import numpy as np
import glob, os, sys
import open3d as o3d

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)
from helper_ply import read_ply
from helper_tool import Plot

ins_colors = [[85, 107, 47],  # ground -> OliveDrab
              [0, 255, 0],  # tree -> Green
              [255, 165, 0],  # building -> orange
              [41, 49, 101],  # Walls ->  darkblue
              [0, 0, 0],  # Bridge -> black
              [0, 0, 255],  # parking -> blue
              [255, 0, 255],  # rail -> Magenta
              [200, 200, 200],  # traffic Roads ->  grey
              [89, 47, 95],  # Street Furniture  ->  DimGray
              [255, 0, 0],  # cars -> red
              [255, 255, 0],  # Footpath  ->  deeppink
              [0, 255, 255],  # bikes -> cyan
              [0, 191, 255]  # water ->  skyblue
             ]

def save_ply_o3d(data, save_name):
    pcd = o3d.geometry.PointCloud()
    # pcd = o3d.cpu.pybind.t.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(data[:, 0:3])
    if np.shape(data)[1] == 3:
        o3d.io.write_point_cloud(save_name, pcd)
    elif np.shape(data)[1] == 6:
        if np.max(data[:, 3:6]) > 20:
            pcd.colors = o3d.utility.Vector3dVector(data[:, 3:6] / 255.)
        else:
            pcd.colors = o3d.utility.Vector3dVector(data[:, 3:6])
        o3d.io.write_point_cloud(save_name, pcd)
    return  
        
if __name__ == '__main__':
    base_dir = '/home/kukdo/data/IEEE_Segmentation/SensatUrban-AttFusion_Cross_PT_MLP_Offset/test_SensatUrban/Log_2023-11-24_12-57-01/test_preds' # pred label path (.label)
    save_dir = '/home/kukdo/data/IEEE_Segmentation/SensatUrban-AttFusion_Cross_PT_MLP_Offset/output' # save path (.ply)
    original_data_dir = '/home/kukdo/data/Workspace/RandLA-SensatUrban/Dataset/SensatUrban/original_block_ply' # original data path (.ply)
    data_path = glob.glob(os.path.join(base_dir, '*.label')) # data path
    data_path = np.sort(data_path)

    visualization = True
    for file_name in data_path:
        # pred_data = read_ply(file_name)
        pred = np.fromfile(file_name, np.uint8)
        # print(pred)
        print("----------------------------------------------")
        print("Loading " + file_name.split('/')[-1][:-6] + " and writing ply...")
        original_data = read_ply(os.path.join(original_data_dir, file_name.split('/')[-1][:-6] + '.ply'))
        # print(original_data)
        labels = original_data['class']
        points = np.vstack((original_data['x'], original_data['y'], original_data['z'])).T

        ##################
        # Visualize data #
        ##################
        if visualization:
            colors = np.vstack((original_data['red'], original_data['green'], original_data['blue'])).T
            xyzrgb = np.concatenate([points, colors], axis=-1)
            GT = Plot.draw_pc_sem_ins(points, labels, ins_colors)  # visualize ground-truth
            # Pred = Plot.draw_pc_sem_ins(points, pred, ins_colors)  # visualize prediction
            save_ply_o3d(GT, os.path.join(save_dir, file_name.split('/')[-1][:-6] + '_GT.ply'))
            # save_ply_o3d(Pred, os.path.join(save_dir, file_name.split('/')[-1][:-6] + '_Pred.ply'))

