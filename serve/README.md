
<div align="center" markdown>
<img src="https://user-images.githubusercontent.com/48245050/182628092-dbe94ed9-a977-473b-9375-ea649204b021.png"/>  

# Serve MMDetection3D

<p align="center">
  <a href="#Overview">Overview</a> •
  <a href="#How-To-Run">How To Run</a> •
  <a href="#How-To-Use">How To Use</a> •
  <a href="#Related-Apps">Related Apps</a> •
  <a href="#Demo">Demo</a> •
  <a href="#Acknowledgment">Acknowledgment</a>
</p>

[![](https://img.shields.io/badge/supervisely-ecosystem-brightgreen)](https://ecosystem.supervise.ly/apps/supervisely-ecosystem/mmdetection_3d/serve)
[![](https://img.shields.io/badge/slack-chat-green.svg?logo=slack)](https://supervise.ly/slack)
![GitHub release (latest SemVer)](https://img.shields.io/github/v/release/supervisely-ecosystem/mmdetection_3d)
[![views](https://app.supervise.ly/img/badges/views/supervisely-ecosystem/mmdetection_3d/serve.png)](https://supervise.ly)
[![runs](https://app.supervise.ly/img/badges/runs/supervisely-ecosystem/mmdetection_3d/serve.png)](https://supervise.ly)

</div>


# Overview

Serve MMDetection3D model as Supervisely Application. MMDetection3D is an open source toolbox based on PyTorch. Learn more about MMDetection3D and available models [here](https://github.com/open-mmlab/mmdetection3d).

Application key points:
- All 3D Detection models from MM Toolbox are available
- Deployed on GPU (NVidia RTX devices support coming soon)
- Models are available only for outdoor and indoor 3d object detection. Other scenarious coming soon (monocular 3D Object Detection, multi-modal 3D Object Detection, 3D Semantic Segmentation)


# How to Run

**Step 1.** Add [Serve MMDetection3D](https://ecosystem.supervise.ly/apps/supervisely-ecosystem/mmdetection_3d/serve) to your team

<img data-key="sly-module-link" data-module-slug="supervisely-ecosystem/mmdetection_3d/serve" src="https://user-images.githubusercontent.com/48913536/171728074-050e43a0-f890-46bb-8f0c-49d11ab6d7c3.png" width="350px" style='padding-bottom: 10px'/>

**Step 2.** Run the application from Plugins & Apps page

<img src="https://user-images.githubusercontent.com/48913536/171728069-8402f8d0-6dff-4ca8-b514-4b7fbc31b9f0.png" width="80%" style='padding-top: 10px'>  

**Step 3.** Press the Run button in the modal window

<img src="https://user-images.githubusercontent.com/48913536/171728060-0cb3c955-5fec-4038-95d3-1085d69457a5.png" width="50%" style='padding-top: 10px'>  

# How to Use

**Pretrained models**

**Step 1.** Select architecture and pretrained model

**Step 2.** Press the **Serve** button

<img src="https://user-images.githubusercontent.com/48913536/171728083-bda7c593-641a-4feb-ace7-4f889cfeedbf.png" width="80%"> 

**Step 3.** Wait for the model to deploy

<img src="https://user-images.githubusercontent.com/48913536/171728064-98ee86fa-841b-4d11-80dd-68cc3411d1ed.png" width="80%">

**Custom models**

Model and directory structure must be acquired via [Train MMDetection](https://app.supervise.ly/ecosystem/apps/mmdetection_3d/train) app or manually created with the same directory structure.

**Step 1.** Open `checkpoints/data` directory in Team Files

<img src="https://user-images.githubusercontent.com/97401023/192815622-9d87b91f-e9a6-4419-93c1-d29f97c438d3.png" width="80%" style='padding-top: 10px'/>  

**Step 2.** Select checkpoint to serve and click right button to open context menu. Select `Copy path`.

<img src="https://user-images.githubusercontent.com/97401023/192815866-8e8683cc-394e-4bd2-aea7-64a5ddf09dae.png" width="80%" style='padding-top: 10px'/>  

**Step 3.** Open [Serve MMDetection3D](https://app.supervise.ly/ecosystem/apps/mmdetection_3d/serve) app and open `Custom weights` tab. Paste checkpoint path to opened text field and press `Serve` button. Your model is ready to use!

<img src="https://user-images.githubusercontent.com/97401023/192815991-e0f70ae7-701e-40ec-9493-f1da57ae443a.png" width="80%" style='padding-top: 10px'/>  

# Related Apps

1. [Train MMDetection3D](https://app.supervise.ly/ecosystem/apps/mmdetection_3d/train) - start training on your custom data. Just run app from the context menu of your project, choose classes of interest, train/val splits, configure training parameters and augmentations, and monitor training metrics in realtime. All training artifacts including model weights will be saved to Team Files and can be easily downloaded. 

    <img data-key="sly-module-link" data-module-slug="supervisely-ecosystem/mmdetection_3d/train" src="https://user-images.githubusercontent.com/97401023/192003567-4446f620-6540-4e68-a6a1-d3a9fcc85fbc.png" width="350px"/>

2. [Serve MMDetection3D](https://app.supervise.ly/ecosystem/apps/mmdetection_3d/serve) - serve model as Rest API service. You can run pretrained model, use custom model weights trained in Supervisely. Thus other apps from Ecosystem can get predictions from the deployed model. Also developers can send inference requiests in a few lines of python code.

    <img data-key="sly-module-link" data-module-slug="supervisely-ecosystem/mmdetection_3d/serve" src="https://user-images.githubusercontent.com/97401023/192003614-4dbe1828-e9c1-4c78-bf89-8f3115103d29.png" width="350px"/>
  
3. [Apply 3D Detection to Pointcloud Project](https://app.supervise.ly/ecosystem/apps/apply-det3d-to-project-dataset) - app allows to play with different inference options and visualize predictions in real time.  Once you choose inference settings you can apply model to all images in your project to visually analise predictions and perform automatic data pre-labeling. 

    <img data-key="sly-module-link" data-module-slug="supervisely-ecosystem/apply-det3d-to-project-dataset" src="https://user-images.githubusercontent.com/97401023/192003658-ec094ea3-2410-470b-b944-cd0a6cc6703b.png" width="550px"/>

4. [Import KITTI 3D](https://app.supervise.ly/ecosystem/apps/import-kitti-3d) - app allows to get sample from KITTI 3D dataset or upload your downloaded KITTI data to Supervisely in point clouds project format.

    <img data-key="sly-module-link" data-module-slug="supervisely-ecosystem/import-kitti-3d" src="https://user-images.githubusercontent.com/97401023/192003697-a6aa11c4-df2e-46cc-9072-b9937756c51b.png" width="350px"/>

5. [Import KITTI-360](https://app.supervise.ly/ecosystem/apps/import-kitti-360/supervisely_app) - app allows to upload your downloaded KITTI-360 data to Supervisely in point cloud episodes project format.

    <img data-key="sly-module-link" data-module-slug="supervisely-ecosystem/import-kitti-360/supervisely_app" src="https://user-images.githubusercontent.com/97401023/192003741-0fd62655-60c3-4e57-80e8-85f936fc0f8d.png" width="350px"/>

# Related Projects

1. [Demo LYFT 3D dataset annotated](https://app.supervise.ly/ecosystem/projects/demo-lyft-3d-dataset-annotated) - demo sample from [Lyft](https://level-5.global/data) dataset with labels.

    <img data-key="sly-module-link" data-module-slug="supervisely-ecosystem/demo-lyft-3d-dataset-annotated" src="https://user-images.githubusercontent.com/97401023/192003812-1cefef97-29e3-40dd-82c6-7d3cf3d55585.png" width="400px"/>

2. [Demo LYFT 3D dataset](https://app.supervise.ly/ecosystem/projects/demo-lyft-3d-dataset) - demo sample from [Lyft](https://level-5.global/data) dataset without labels.

    <img data-key="sly-module-link" data-module-slug="supervisely-ecosystem/demo-lyft-3d-dataset" src="https://user-images.githubusercontent.com/97401023/192003862-102de613-d365-4043-8ca0-d59e3c95659a.png" width="400px"/>

3. [Demo KITTI pointcloud episodes annotated](https://app.supervise.ly/ecosystem/projects/demo-kitti-3d-episodes-annotated) - demo sample from [KITTI 3D](https://www.cvlibs.net/datasets/kitti/eval_tracking.php) dataset with labels.

    <img data-key="sly-module-link" data-module-slug="supervisely-ecosystem/demo-kitti-3d-episodes-annotated" src="https://user-images.githubusercontent.com/97401023/192003917-71425add-e985-4a9c-8739-df832324be2f.png" width="400px"/>

4. [Demo KITTI pointcloud episodes](https://app.supervise.ly/ecosystem/projects/demo-kitti-3d-episodes) - demo sample from [KITTI 3D](https://www.cvlibs.net/datasets/kitti/eval_tracking.php) dataset without labels.

    <img data-key="sly-module-link" data-module-slug="supervisely-ecosystem/demo-kitti-3d-episodes" src="https://user-images.githubusercontent.com/97401023/192003975-972c1803-b502-4389-ae83-72958ddd89ad.png" width="400px"/>


# Demo video

<a data-key="sly-embeded-video-link" href="https://youtu.be/wh5bwPWSvXw" data-video-code="wh5bwPWSvXw"> <img src="https://user-images.githubusercontent.com/48913536/171728077-4905ba22-a0fc-43df-8026-0b8d3e9f53dc.png" alt="SLY_EMBEDED_VIDEO_LINK"  width="500"> </a>  


# Acknowledgment

This app is based on the great work `MMDetection3D` ([github](https://github.com/open-mmlab/mmdetection3d)). ![GitHub Org's stars](https://img.shields.io/github/stars/open-mmlab/mmdetection3d?style=social)

