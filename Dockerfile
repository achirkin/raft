FROM nvidia/cuda:12.0.0-devel-ubuntu22.04
MAINTAINER Artem Chirkin <achirkin@nvidia.com>

### Setup some basic environment settings
ENV SHELL=/bin/bash \
    TERM=linux \
    DEBIAN_FRONTEND=noninteractive

### Setup up the locale
RUN apt-get update \
 && apt-get install --no-install-recommends -y apt-utils \
 && apt-get install --no-install-recommends -y locales \
 && locale-gen en_US.UTF-8 \
 && echo "LC_ALL=en_US.UTF-8" >> /etc/environment \
 && echo "en_US.UTF-8 UTF-8" >> /etc/locale.gen \
 && echo "LANG=en_US.UTF-8" > /etc/locale.conf
ENV LANGUAGE=en_US.UTF-8 \
    LANG=en_US.UTF-8 \
    LC_ALL=en_US.UTF-8
RUN update-locale

### Install distribution dependencies
RUN echo 'deb http://ports.ubuntu.com/ubuntu-ports/ jammy-proposed main restricted universe multiverse' >> /etc/apt/sources.list
RUN apt-get update \
 && apt-get upgrade -y \
 && apt-get dist-upgrade -y \
 && apt-get autoclean -y \
 && apt-get autoremove -y
RUN apt-get install --no-install-recommends -y \
      lsb-release sudo wget bzip2 ca-certificates curl git git-lfs libxkbfile-dev libsecret-1-dev \
      openssl net-tools dumb-init bash-completion nano ccache ssh less
RUN apt-get install --no-install-recommends -y \
      libblas-dev liblapack-dev
RUN apt-get install --no-install-recommends -y \
      clang-15 clang-tools-15 clang-tidy-15 ninja-build python3.10 gcc-12 g++-12

RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-12 30 ; \
    update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 20 ; \
    update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-12 30 ; \
    update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-11 20 ; \
    update-alternatives --install /usr/bin/cc cc /usr/bin/gcc 30 ; \
    update-alternatives --set cc /usr/bin/gcc ; \
    update-alternatives --install /usr/bin/c++ c++ /usr/bin/g++ 30 ; \
    update-alternatives --set c++ /usr/bin/g++

### Install cmake
RUN wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - | sudo tee /usr/share/keyrings/kitware-archive-keyring.gpg >/dev/null
RUN echo "deb [signed-by=/usr/share/keyrings/kitware-archive-keyring.gpg] https://apt.kitware.com/ubuntu/ `lsb_release -sc` main" | sudo tee /etc/apt/sources.list.d/kitware.list >/dev/null
RUN apt-get update && apt-get install -y cmake

### Profiling: install nsys
RUN wget -qO - https://developer.download.nvidia.com/devtools/repos/ubuntu2204/amd64/nvidia.pub | gpg --dearmor > /etc/apt/trusted.gpg.d/nvidia-devtools.gpg
RUN echo "deb https://developer.download.nvidia.com/devtools/repos/ubuntu2204/\$(ARCH) /" > /etc/apt/sources.list.d/nvidia-devtools.list
RUN apt-get update && apt-get install -y nsight-compute nsight-systems

### Symlink executables
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 10
# RUN update-alternatives --install /usr/bin/ncu ncu `ls /opt/nvidia/nsight-compute/*/ncu` 10
RUN update-alternatives --install /usr/bin/nsys nsys `ls /opt/nvidia/nsight-systems/*/bin/nsys` 10

### Clone and build raft
RUN umask 0000 && mkdir /workspace
COPY . /workspace/raft

### Done! run the init job and code-server
ENTRYPOINT ["/bin/bash"]
