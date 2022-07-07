# use nvidia cuda/cudnn image with miniconda on top
FROM gpuci/miniconda-cuda:11.3-devel-ubuntu18.04

# update GPG key and install linux development CLI tools
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub \
    && apt update \
    && apt install -y \
         git \
         make \
	 sed \
         vim \
         wget

# allow history search in terminal
RUN echo "\"\e[A\": history-search-backward" > ~/.inputrc
RUN echo "\"\e[B\": history-search-forward" ~/.inputrc

# install core Python environment and system packages
COPY ./Makefile ./environment.yml /
RUN make conda-update

# switch to a login shell after cleaning up error-causing line in /root/.profile, see https://www.educative.io/answers/error-mesg-ttyname-failed-inappropriate-ioctl-for-device
RUN sed -i "s/mesg n || true/tty -s \&\& mesg n/" $HOME/.profile
SHELL ["conda", "run", "--no-capture-output", "-n", "fsdl-text-recognizer-2022", "/bin/bash", "-c"]

# install the core requirements, then remove build files
COPY requirements /requirements
RUN make pip-tools && rm -rf /Makefile /requirements

# add current dir to PYTHONPATH so libraries are importable
ENV PYTHONPATH=.:$PYTHONPATH
