#!/usr/bin/env python3

import rospy
from pcdet.utils import common_utils
from pcdet.models import build_network, load_data_to_gpu
from pcdet.datasets import DatasetTemplate
from pcdet.config import cfg, cfg_from_yaml_file
import torch
import numpy as np
from std_msgs.msg import Header
from sensor_msgs.msg import PointCloud2
import argparse
import glob
from pathlib import Path

import sys
sys.path.append('../tools')

try:
    import open3d
    from visual_utils import open3d_vis_utils as V
    OPEN3D_FLAG = True
except:
    import mayavi.mlab as mlab
    from visual_utils import visualize_utils as V
    OPEN3D_FLAG = False


class DemoDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None, ext='.bin'):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        """
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.root_path = root_path
        self.ext = ext
        data_file_list = glob.glob(
            str(root_path / f'*{self.ext}')) if self.root_path.is_dir() else [self.root_path]

        data_file_list.sort()
        self.sample_file_list = data_file_list

    def __len__(self):
        return len(self.sample_file_list)

    def __getitem__(self, index):
        if self.ext == '.bin':
            points = np.fromfile(
                self.sample_file_list[index], dtype=np.float32).reshape(-1, 4)
        elif self.ext == '.npy':
            points = np.load(self.sample_file_list[index])
        else:
            raise NotImplementedError

        input_dict = {
            'points': points,
            'frame_id': index,
        }

        data_dict = self.prepare_data(data_dict=input_dict)
        return data_dict


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default='cfgs/kitti_models/second.yaml',
                        help='specify the config for demo')
    # parser.add_argument('--data_path', type=str, default='demo_data',
    #                     help='specify the point cloud data file or directory')
    parser.add_argument('--ckpt', type=str, default=None,
                        help='specify the pretrained model')
    parser.add_argument('--ext', type=str, default='.bin',
                        help='specify the extension of your point cloud data file')

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)

    return args, cfg


def main():
    print(OPEN3D_FLAG)
    args, cfg = parse_config()
    logger = common_utils.create_logger()
    logger.info(
        '-----------------Quick Demo of OpenPCDet-------------------------')

    rospy.init_node('OpenPCDet_demo', anonymous=True)

    save_path = './pointcloud/temp.npy'

    demo_dataset = DemoDataset(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
        root_path=Path(save_path), ext=args.ext, logger=logger
    )

    logger.info('Total number of samples: \t{}'.format(len(demo_dataset)))

    model = build_network(model_cfg=cfg.MODEL, num_class=len(
        cfg.CLASS_NAMES), dataset=demo_dataset)
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)
    model.cuda()
    model.eval()

    vis = open3d.visualization.Visualizer()
    vis.create_window()

    while not rospy.is_shutdown():
        # Load saved point cloud frame
        points = np.load(save_path)

        input_dict = {
            'points': points,
            'frame_id': 0,  # Assuming frame ID is 0 for simplicity
        }

        data_dict = demo_dataset.prepare_data(data_dict=input_dict)
        data_dict = demo_dataset.collate_batch([data_dict])
        load_data_to_gpu(data_dict)

        with torch.no_grad():
            pred_dicts, _ = model.forward(data_dict)

        # Visualize results using Open3D

        vis.clear_geometries()

        points = data_dict['points'][:, 1:]
        gt_boxes = None
        ref_boxes = pred_dicts[0]['pred_boxes']
        ref_scores = pred_dicts[0]['pred_scores']
        ref_labels = pred_dicts[0]['pred_labels']
        draw_origin = True
        point_colors = None

        if isinstance(points, torch.Tensor):
            points = points.cpu().numpy()
        if isinstance(gt_boxes, torch.Tensor):
            gt_boxes = gt_boxes.cpu().numpy()
        if isinstance(ref_boxes, torch.Tensor):
            ref_boxes = ref_boxes.cpu().numpy()

        vis.get_render_option().point_size = 1.0
        vis.get_render_option().background_color = np.zeros(3)

        # draw origin
        if draw_origin:
            axis_pcd = open3d.geometry.TriangleMesh.create_coordinate_frame(
                size=1.0, origin=[0, 0, 0])
            vis.add_geometry(axis_pcd)

        pts = open3d.geometry.PointCloud()
        pts.points = open3d.utility.Vector3dVector(points[:, :3])

        vis.add_geometry(pts)
        if point_colors is None:
            pts.colors = open3d.utility.Vector3dVector(
                np.ones((points.shape[0], 3)))
        else:
            pts.colors = open3d.utility.Vector3dVector(point_colors)

        if gt_boxes is not None:
            vis = V.draw_box(vis, gt_boxes, (0, 0, 1))

        if ref_boxes is not None:
            vis = V.draw_box(vis, ref_boxes, (0, 1, 0), ref_labels, ref_scores)

        vis.run()
        # vis.destroy_window()

        # if not OPEN3D_FLAG:
        #     mlab.show(stop=True)

    logger.info('Demo done.')


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
