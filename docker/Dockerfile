FROM supervisely/base-py-sdk:6.15.0

RUN pip3 install -U torch==1.11.0+cu115 torchvision==0.12.0+cu115 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip3 install mmcv-full==1.5.0 -f https://download.openmmlab.com/mmcv/dist/cu115/torch1.11.0/index.html -U

# Temporary solution to install mmdetection3d from source.
# TODO: use pip3 install mmdet3d==1.0.0 when it will be released
RUN git clone https://github.com/open-mmlab/mmdetection3d.git /tmp/mmdet3d/mmdetection3d
RUN pip3 install -e /tmp/mmdet3d/mmdetection3d
RUN pip3 install mmdet==2.23.0
RUN pip3 install mmsegmentation==0.23.0
RUN pip3 install pyyaml==6.0
# It is possible that not all necessary packages specified here

#RUN mkdir -p /tmp/mmdet3d \
#    && wget https://github.com/open-mmlab/mmdetection3d/archive/refs/tags/v1.0.0rc1.zip -P /tmp/mmdet3d
