FROM pytorch/pytorch:1.5-cuda10.1-cudnn7-runtime

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y gnupg2
RUN apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-keys E1DD270288B4E6030699E45FA1715D88E1DF1F24

RUN apt-get update -y
RUN apt-get -y install ffmpeg libsm6 libxext6 gcc mono-mcs
RUN apt-get install git python3-pip -y

RUN python3 -m pip install torch==1.5 torchvision==0.6 -f https://download.pytorch.org/whl/cu101/torch_stable.html
RUN python3 -m pip install cython pyyaml==5.1 --ignore-installed
RUN python3 -m pip install git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI
RUN python3 -m pip install detectron2==0.1.3 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu101/torch1.5/index.html

ADD requirements.txt .
RUN python3 -m pip install -r requirements.txt

ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility