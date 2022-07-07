# use nvidia's barebones cuda/cudnn image
FROM nvidia/cuda:11.1-cudnn8-runtime-ubuntu18.04

# update GPG key and install linux development CLI tools
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub \
    && apt update \
    && apt install -y \
         git \
         make \
         vim \
         wget
         
# allow history search in terminal
RUN echo "\"\e[A\": history-search-backward" > ~/.inputrc
RUN echo "\"\e[B\": history-search-forward" ~/.inputrc

# install miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-py37_4.12.0-Linux-x86_64.sh -O ~/miniconda.sh \
    && bash ~/miniconda.sh -b -p $HOME/miniconda \
    && /root/miniconda/condabin/conda init
    
# use a login bash shell so that conda is accessible, see https://pythonspeed.com/articles/activate-conda-dockerfile
SHELL ["/bin/bash", "--login", "-c"]

# install core Python environment and system packages
RUN make conda-update

# add conda activate to the .bashrc so it's active in login bash shells
RUN echo "conda activate fsdl-text-recognizer-2022" > ~/.bashrc

# install the core requirements
RUN make pip-tools

# add current dir to PYTHONPATH so libraries are importable
ENV PYTHONPATH=.:$PYTHONPATH
