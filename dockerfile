FROM nvidia/cuda:12.6.3-devel-ubuntu22.04

USER root

RUN useradd -u 1000 -m user --shell /bin/bash \
   && mkdir -p /home/user \
   && chown user:user /home/user \
   && mkdir -p /home/jovyan \
   && chown user:user /home/jovyan

# Installing the necessary packages
# Additional packages can be installed
RUN apt-get update \
   && apt-get install -y --no-install-recommends \
   openssh-server openssh-client \
   curl wget bzip2 \
   # python3 python3-pip \
   && rm -rf /var/lib/apt/lists/*


# Adding support for Openmp (optional)
# For all images except NVIDIA NGC, which do not require the installation of HPC-X
ARG UBUNTU_VER=22.04
ARG HPCX_VER=2.18.1
ARG HPCX_DIR=/opt/hpcx
ARG HPCX_PACKAGE=hpcx-v${HPCX_VER}-gcc-mlnx_ofed-ubuntu${UBUNTU_VER}-cuda12-x86_64.tbz

RUN mkdir -p ${HPCX_DIR} \
   && wget --quiet --no-check-certificate -c https://content.mellanox.com/hpc/hpc-x/v${HPCX_VER}/${HPCX_PACKAGE} -O /tmp/${HPCX_PACKAGE} \
   && tar xjf /tmp/${HPCX_PACKAGE} -C ${HPCX_DIR} --strip-components=1

RUN apt-get clean

# Setting the environment variables
# To save environment variables correctly, we recommend creating a profile file in "/etc/profile.d/mlspace.sh",
# which will contain the necessary environment variables that are activated when working in the image

ENV LD_LIBRARY_PATH=/opt/hpcx/ompi/lib:/opt/hpcx/ucx/lib:/opt/hpcx/ucc/lib:/opt/hpcx/sharp/lib:/opt/hpcx/nccl_rdma_sharp_plugin/lib:/opt/hpcx/hcoll/lib:${LD_LIBRARY_PATH}
ENV PATH=/opt/hpcx/ompi/bin:/opt/hpcx/ucx/bin:/opt/hpcx/ucc/bin:/opt/hpcx/sharp/bin:/opt/hpcx/hcoll/bin:${PATH}
ENV OPAL_PREFIX=/opt/hpcx/ompi

RUN echo "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH">/etc/profile.d/mlspace.sh \
   && echo "export PATH=$PATH" >> /etc/profile.d/mlspace.sh \
   && echo "export OPAL_PREFIX=$OPAL_PREFIX" >> /etc/profile.d/mlspace.sh

ENV CUDA_HOME=/usr/local/cuda-12.6 \
    PYTORCH_INDEX_URL=https://download.pytorch.org/whl/nightly/cu126


RUN apt-get update && apt-get install -y --no-install-recommends \
   python3 python3-pip python3-venv python3-dev git\
   && rm -rf /var/lib/apt/lists/*

RUN pip3 install --upgrade pip

# Install PyTorch with CUDA 12.6
RUN pip3 install --pre torch torchvision torchaudio --index-url $PYTORCH_INDEX_URL

# Install additional packages for building extensions
RUN pip3 install packaging ninja wheel setuptools setuptools-scm

# Install requirements.txt packages directly
RUN pip3 install \
    einops \
    tqdm \
    coolname \
    pydantic \
    argdantic \
    wandb \
    clearml \
    numpy \
    pandas \
    omegaconf \
    hydra-core \
    huggingface_hub \
    numba \
    triton

RUN pip3 install --no-build-isolation adam-atan2

RUN pip3 install --upgrade 'setuptools>=61'

RUN git clone https://github.com/Dao-AILab/flash-attention.git /tmp/flash-attention
WORKDIR /tmp/flash-attention/hopper
RUN MAX_JOBS=8 python3 setup.py install

# Clean up temporary files
RUN rm -rf /tmp/flash-attention

RUN ln -s /usr/bin/python3 /usr/bin/python

USER user

# Setting the necessary environment variables on behalf of an unprivileged user
ENV LD_LIBRARY_PATH=/opt/hpcx/ompi/lib:/opt/hpcx/ucx/lib:/opt/hpcx/ucc/lib:/opt/hpcx/sharp/lib:/opt/hpcx/nccl_rdma_sharp_plugin/lib:/opt/hpcx/hcoll/lib:${LD_LIBRARY_PATH}
ENV PATH=/opt/hpcx/ompi/bin:/opt/hpcx/ucx/bin:/opt/hpcx/ucc/bin:/opt/hpcx/sharp/bin:/opt/hpcx/hcoll/bin:${PATH}
ENV OPAL_PREFIX=/opt/hpcx/ompi

# Actions for the final stage
SHELL ["/bin/bash", "-cu"]