
<div align="center" markdown>

<img src=""/>  

# Train MMDetection

<p align="center">
  <a href="#Overview">Overview</a> •
  <a href="#How-To-Run">How To Run</a> •
  <a href="#Training-on-Demo-Data">TTraining on Demo Data</a> •
  <a href="#Related-Apps">Related Apps</a> •
  <a href="#Related-Projects">Related Projects</a> •
  <a href="#Screenshot">Screenshot</a> •
  <a href="#Acknowledgment">Acknowledgment</a>
</p>

[![](https://img.shields.io/badge/supervisely-ecosystem-brightgreen)](https://ecosystem.supervise.ly/apps/supervisely-ecosystem/mmdetection_3d/train)
[![](https://img.shields.io/badge/slack-chat-green.svg?logo=slack)](https://supervise.ly/slack)
![GitHub release (latest SemVer)](https://img.shields.io/github/v/release/supervisely-ecosystem/mmdetection_3d)
[![views](https://app.supervise.ly/img/badges/views/supervisely-ecosystem/mmdetection_3d/train.png)](https://supervise.ly)
[![runs](https://app.supervise.ly/img/badges/runs/supervisely-ecosystem/mmdetection_3d/train.png)](https://supervise.ly)

</div>

# Overview

Train MMDetection3D models in Supervisely.

Application key points:
- Object Detection 3D models from MM Toolbox
- Use pretrained MMDetection3D models
- Work with point clouds of any size and shape
- Define Train / Validation splits
- Select classes for training
- Define augmentations
- Tune hyperparameters
- Monitor Metric charts
- Save training artifacts to Team Files

**Supported MMDetection3D models [(model zoo)](https://github.com/open-mmlab/mmdetection3d/blob/master/docs/en/model_zoo.md):**

<div align="center">
  <b>Architectures</b>
</div>
<table align="center">
  <tbody>
    <tr align="center" valign="bottom">
      <td>
        <b>Outdoor 3D Detection (lidar data only)</b>
      </td>
    </tr>
    <tr valign="top">
      <td>
        <ul>
          <li><a href="configs/centerpoint">CenterPoint (CVPR'2021)</a></li>
        </ul>
      </td>
    </tr>
</td>
    </tr>
  </tbody>
</table>

# How to Run

Run app [Train MMDetection3D](https://app.supervise.ly/ecosystem/apps/mmdetection_3d/train) from ecosystem or from context menu of the point cloud / point cloud episodes project with annotations (`Cuboid3D` is supported as label type for object detection 3D)

<img data-key="sly-module-link" data-module-slug="supervisely-ecosystem/mmdetection_3d/train" src="" width="350px" style='padding-bottom: 10px'/>


# Training on Demo Data

You can try training on demo data sample. Set following settings in training dashboard:

- `data`: [Demo Lyft 3D dataset annotated](https://app.supervise.ly/ecosystem/projects/demo-lyft-3d-dataset-annotated)
- `Model`: centerpoint_0075voxel_second_secfpn_dcn_circlenms_4x8_cyclic_20e_nus
- `Classes`: Car
- `Train / val splits`: 100 / 25
- `Data preparation`: sliding window (x: 108, y: 108, z: 15)
- `Augmentations`: No Augmentations
- `Training hyperparams`: default

[gif here]


# Related Apps

1. [Train MMDetection3D](https://app.supervise.ly/ecosystem/apps/mmdetection_3d/train) - start training on your custom data. Just run app from the context menu of your project, choose classes of interest, train/val splits, configure training parameters and augmentations, and monitor training metrics in realtime. All training artifacts including model weights will be saved to Team Files and can be easily downloaded. 

    <img data-key="sly-module-link" data-module-slug="supervisely-ecosystem/mmdetection_3d/train" src="" width="350px"/>

2. [Serve MMDetection3D](https://app.supervise.ly/ecosystem/apps/mmdetection_3d/serve) - serve model as Rest API service. You can run pretrained model, use custom model weights trained in Supervisely. Thus other apps from Ecosystem can get predictions from the deployed model. Also developers can send inference requiests in a few lines of python code.

    <img data-key="sly-module-link" data-module-slug="supervisely-ecosystem/mmdetection_3d/serve" src="" width="350px"/>
  
3. [Apply 3D Detection to Pointcloud Project](https://app.supervise.ly/ecosystem/apps/apply-det3d-to-project-dataset) - app allows to play with different inference options and visualize predictions in real time.  Once you choose inference settings you can apply model to all images in your project to visually analise predictions and perform automatic data pre-labeling. 

    <img data-key="sly-module-link" data-module-slug="supervisely-ecosystem/apply-det3d-to-project-dataset" src="" width="350px"/>

4. [Import KITTI 3D](https://app.supervise.ly/ecosystem/apps/import-kitti-3d) - app allows to get sample from KITTI 3D dataset or upload your downloaded KITTI data to Supervisely in point clouds project format.

  <img data-key="sly-module-link" data-module-slug="supervisely-ecosystem/import-kitti-3d" src="" width="350px"/>

5. [Import KITTI-360](https://app.supervise.ly/ecosystem/apps/import-kitti-360/supervisely_app) - app allows to upload your downloaded KITTI-360 data to Supervisely in point cloud episodes project format.

    <img data-key="sly-module-link" data-module-slug="supervisely-ecosystem/import-kitti-360/supervisely_app" src="" width="350px"/>

# Related Projects

1. [Demo LYFT 3D dataset annotated](https://app.supervise.ly/ecosystem/projects/demo-lyft-3d-dataset-annotated) - demo sample from [Lyft](https://level-5.global/data) dataset with labels.

    <img data-key="sly-module-link" data-module-slug="supervisely-ecosystem/demo-lyft-3d-dataset-annotated" src="" width="350px"/>

2. [Demo LYFT 3D dataset](https://app.supervise.ly/ecosystem/projects/demo-lyft-3d-dataset) - demo sample from [Lyft](https://level-5.global/data) dataset without labels.

    <img data-key="sly-module-link" data-module-slug="supervisely-ecosystem/demo-lyft-3d-dataset" src="" width="350px"/>

3. [Demo KITTI pointcloud episodes annotated](https://app.supervise.ly/ecosystem/projects/demo-kitti-3d-episodes-annotated) - demo sample from [KITTI 3D](https://www.cvlibs.net/datasets/kitti/eval_tracking.php) dataset with labels.

    <img data-key="sly-module-link" data-module-slug="supervisely-ecosystem/demo-kitti-3d-episodes-annotated" src="" width="350px"/>

4. [Demo KITTI pointcloud episodes](https://app.supervise.ly/ecosystem/projects/demo-kitti-3d-episodes) - demo sample from [KITTI 3D](https://www.cvlibs.net/datasets/kitti/eval_tracking.php) dataset without labels.

    <img data-key="sly-module-link" data-module-slug="supervisely-ecosystem/demo-kitti-3d-episodes" src="" width="350px"/>

# Screenshot

<img src="" width="100%" style='padding-top: 10px'>

# Acknowledgment

This app is based on the great work `MMDetection3D` ([github](https://github.com/open-mmlab/mmdetection3d)). ![GitHub Org's stars](https://img.shields.io/github/stars/open-mmlab/mmdetection3d?style=social)
