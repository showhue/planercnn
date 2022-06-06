FROM nvidia/cuda:9.2-cudnn7-devel-ubuntu18.04
MAINTAINER jason "jason4775@gmail.com"

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    build-essential && \
    rm -rf /var/lib/apt/lists/*

RUN curl -o ~/miniconda.sh -O https://repo.anaconda.com/miniconda/Miniconda3-4.3.11-Linux-x86_64.sh && \
    chmod +x ~/miniconda.sh && \
    ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh
ENV PATH /opt/conda/bin:$PATH

RUN conda install python=3.6 jupyter pip=9.0.1
RUN conda clean -ya
RUN echo "export PATH=/opt/conda/bin:\$PATH" > /etc/profile.d/conda.sh

RUN pip install http://download.pytorch.org/whl/cu92/torch-0.4.1-cp36-cp36m-linux_x86_64.whl
RUN pip install --upgrade pip
RUN rm -rf ~/.cache/pip

COPY requirements.txt /tmp
RUN pip install -r /tmp/requirements.txt
RUN apt update && apt install -y libsm6 libxext6 libxrender-dev

CMD [ "/bin/bash" ]
