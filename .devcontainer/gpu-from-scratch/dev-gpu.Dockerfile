# use nvidia cuda/cudnn image with miniconda on top
FROM gpuci/miniconda-cuda:11.3-devel-ubuntu18.04

# update GPG key and install linux development CLI tools
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub \
    && apt update \
    && apt install -y \
         git \
         make \
	 sed \
	 tmux \
         vim \
         wget

# allow history search in terminal
RUN echo "\"\e[A\": history-search-backward" > $HOME/.inputrc && echo "\"\e[B\": history-search-forward" $HOME/.inputrc

# move into the root user's home directory
WORKDIR /root

# install core Python environment and system packages
COPY ./Makefile ./environment.yml ./
RUN make conda-update

# switch to a login shell after cleaning up config:
#   removing error-causing line in /root/.profile, see https://www.educative.io/answers/error-mesg-ttyname-failed-inappropriate-ioctl-for-device
#   removing environment-setting in /root/.bashrc
RUN sed -i "s/mesg n || true/tty -s \&\& mesg n/" $HOME/.profile
RUN sed -i "s/conda activate base//" $HOME/.bashrc
SHELL ["conda", "run", "--no-capture-output", "-n", "fsdl-text-recognizer-2022", "/bin/bash", "-c"]

# install the core requirements, then remove build files
COPY ./requirements ./requirements
RUN make pip-tools && rm -rf ./Makefile ./requirements ./environment.yml

# add current dir to PYTHONPATH so libraries are importable
ENV PYTHONPATH=.:$PYTHONPATH

# run all commands inside the conda environment
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "fsdl-text-recognizer-2022", "/bin/bash"]
