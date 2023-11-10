FROM supervisely/base-py-sdk:6.72.66

RUN pip3 install -U torch==1.11.0+cu113 torchvision==0.12.0+cu113 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip3 install mmcv-full==1.6.1 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.11.0/index.html -U

RUN git clone -b v1.0.0rc4 https://github.com/open-mmlab/mmdetection3d.git /tmp/mmdet3d/mmdetection3d
RUN pip3 install -e /tmp/mmdet3d/mmdetection3d
RUN pip3 install mmdet==2.25.1
RUN pip3 install mmsegmentation==0.27.0
RUN pip3 install pyyaml==6.0
RUN pip3 install open3d==0.13.0
RUN pip3 install spconv-cu113

RUN apt-get -y install wget

RUN pip3 install numpy==1.23.5
RUN pip3 uninstall -y numba
RUN pip3 install -U numba
RUN pip3 install trimesh==3.11.2
RUN pip3 install protobuf==3.20.3
RUN pip3 install executing==1.1.1