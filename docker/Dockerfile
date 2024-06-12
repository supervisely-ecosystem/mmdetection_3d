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
RUN pip3 install setuptools==69.5.1

COPY dev_requirements.txt dev_requirements.txt
RUN pip3 install -r dev_requirements.txt
