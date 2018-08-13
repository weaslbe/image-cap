FROM nvidia/cuda:9.0-cudnn7-runtime-ubuntu16.04

ARG USERNAME=docker

RUN apt-get update

RUN apt-get install -y python3 && apt-get install -y python3-pip && apt-get install -y git && apt-get install -y wget

ADD requirements.txt /repo/requirements.txt

RUN python3 -m pip --no-cache-dir install -r /repo/requirements.txt

RUN git clone https://github.com/facebookresearch/fastText.git

RUN cd fastText && make && python3 -m pip --no-cache-dir install .

ENV CUDA_HOME=/usr/local/cuda
ENV CUDA_ROOT=$CUDA_HOME
ENV PATH=$PATH:$CUDA_ROOT/bin:$HOME/bin
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CUDA_ROOT/lib64

RUN git clone --single-branch -b pretrained_embeddings  https://github.com/weaslbe/image-cap.git

RUN wget -nv https://www.dropbox.com/s/pi0v32kzf5ogfm0/submission.hdf5?dl=0

RUN mv submission.hdf5?dl=0 image-cap/submission_weights.hdf5

RUN wget -nv https://www.dropbox.com/s/fbb96qcdpofae84/parameters.json?dl=0

RUN mv parameters.json?dl=0 image-cap/parameters.json

RUN wget -nv https://www.dropbox.com/s/uc6pxmsaid1cgp9/rev_word_index.json?dl=0

RUN mv rev_word_index.json?dl=0 image-cap/rev_word_index.json

WORKDIR /image-cap
