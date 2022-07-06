# use nvidia's barebones cuda/cudnn image
FROM nvidia/cuda:11.1-cudnn8-runtime-ubuntu18.04

# update GPG key and install wget
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub \
    && apt update \
    && apt install -y \
         wget \
         make
   
# install miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-py37_4.12.0-Linux-x86_64.sh -O ~/miniconda.sh \
    && bash ~/miniconda.sh -b -p $HOME/miniconda \
    && /root/miniconda/condabin/conda init

# path management
ENV PATH=/root/miniconda/condabin:$PATH
ENV PYTHONPATH=.:$PYTHONPATH
