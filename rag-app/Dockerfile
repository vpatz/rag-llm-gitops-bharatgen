FROM registry.access.redhat.com/ubi8/python-311

ENV PYTHONUNBUFFERED=1
ENV FORCE_CUDA=0
ENV TORCH_CUDA_ARCH_LIST=""

USER root

# System deps
RUN dnf install -y \
        git \
        gcc \
        gcc-c++ \
        make \
        curl \
    && dnf clean all

WORKDIR /opt/app-root/src

# Python deps (system python)
RUN /usr/bin/python3 -m pip install --upgrade pip

RUN /usr/bin/python3 -m pip install --no-cache-dir \
    torch torchvision \
    --index-url https://download.pytorch.org/whl/cpu

RUN /usr/bin/python3 -m pip install --no-cache-dir \
    flask \
    docling \
    pdfplumber \
    rapidfuzz \
    requests \
    pymilvus \
    joblib \
    tqdm \
    scipy \
    scikit-learn \
    sentence_splitter \
    cohere \
    gradio \
    numpy==1.26.4

# -----------------------------
# ENTRYPOINT: THIS IS THE KEY
# -----------------------------
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/bin/sh", "/entrypoint.sh"]
