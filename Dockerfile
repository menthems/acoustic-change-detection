# Base image
FROM nvidia/cuda:11.8.0-devel-ubuntu20.04
#FROM nvidia/cudagl:10.1-devel-ubuntu16.04 
#(in the Dockerfile on the SS github, outdated for the soundspaces step)

# Setup basic packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    vim \
    ca-certificates \
    libjpeg-dev \
    libpng-dev \
    libglfw3-dev \
    libglm-dev \
    libx11-dev \
    libomp-dev \
    libegl1-mesa-dev \
    libsndfile1 \
    libxcursor-dev \
    libxi-dev \
    libxinerama-dev \
    libxrandr-dev \
    pkg-config \
    wget \
    zip \
    unzip &&\
    rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN curl -sL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o /tmp/miniconda.sh && \
    chmod +x /tmp/miniconda.sh && \
    /tmp/miniconda.sh -b -p /opt/conda && \
    rm /tmp/miniconda.sh && \
    /opt/conda/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    /opt/conda/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r && \
    /opt/conda/bin/conda install -y numpy pyyaml scipy ipython mkl mkl-include && \
    /opt/conda/bin/conda clean -ya

ENV PATH=/opt/conda/bin:$PATH


# Install cmake
RUN wget https://github.com/Kitware/CMake/releases/download/v3.14.0/cmake-3.14.0-Linux-x86_64.sh
RUN mkdir /opt/cmake
RUN sh /cmake-3.14.0-Linux-x86_64.sh --prefix=/opt/cmake --skip-license
RUN ln -s /opt/cmake/bin/cmake /usr/local/bin/cmake
RUN cmake --version

# Conda environment
RUN conda create -n soundspaces python=3.9 cmake=3.14.0

# Setup habitat-sim
#RUN git clone --branch stable https://github.com/facebookresearch/habitat-sim.git
# (stable is V0.1.7; however modified to be the same as on the github installation from SS)
RUN git clone --branch RLRAudioPropagationUpdate https://github.com/facebookresearch/habitat-sim.git
RUN /bin/bash -c ". activate soundspaces; cd habitat-sim; pip install -r requirements.txt; python setup.py build_ext --parallel 1 install --headless --audio"
# (parallel option is necessary to avoid OOM error)

# Install challenge specific habitat-lab
#RUN git clone --branch stable https://github.com/facebookresearch/habitat-lab.git
# (stable is V0.1.7; however modified to be the same as on the github installation from SS)
RUN git clone --branch v0.2.2 https://github.com/facebookresearch/habitat-lab.git
RUN /bin/bash -c ". activate soundspaces; cd habitat-lab; git checkout v0.1.6; pip install -e ."

# Install challenge specific habitat-lab
RUN pwd
RUN git clone --branch main https://github.com/facebookresearch/sound-spaces.git
RUN /bin/bash -c ". activate soundspaces; cd sound-spaces;pip install -e . -v"
# (needs U20, else build error with soxr, ml_dtypes & optree)

# Silence habitat-sim logs
ENV GLOG_minloglevel=2
ENV MAGNUM_LOG="quiet"

#(!! error with linking ("Relink `.../libRLRAudioPropagation.so' with `/lib/x86_64-linux-gnu/libz.so.1' for IFUNC symbol `crc32_z' free(): invalid pointer") can be solved by importing habitat before importing habitat_sim)
