FROM ubuntu:latest
ARG PYTHON_VERSION=3.10.12
ARG USERNAME=devuser
ARG USER_UID=1001
ARG USER_GID=$USER_UID
ENV TZ=Asia/Tokyo \
    LC_ALL=C.UTF-8 \
    LANG=C.UTF-8 \
    PYTHONIOENCODING=utf-8 \
    DEBIAN_FRONTEND=noninteractive

WORKDIR /tmp/

# Install base packages including ca-certificates
RUN apt-get update \
    && apt-get install -y ca-certificates \
    && update-ca-certificates \
    && apt-get install -y --no-install-recommends \
        wget git zip curl make llvm tzdata tk-dev graphviz xz-utils zlib1g-dev \
        libssl-dev libbz2-dev libffi-dev liblzma-dev libsqlite3-dev libgl1-mesa-dev \
        libreadline-dev libncurses5-dev libncursesw5-dev build-essential nano git-lfs unzip\
    && git lfs install \
    && rm -rf /var/lib/apt/lists/*

# Set timezone
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Install Python from source
RUN wget https://www.python.org/ftp/python/$PYTHON_VERSION/Python-$PYTHON_VERSION.tar.xz \
    && tar -xf Python-$PYTHON_VERSION.tar.xz \
    && cd Python-$PYTHON_VERSION \
    && ./configure --enable-optimizations \
    && make -j$(nproc) install \
    && cd .. \
    && rm -rf Python-$PYTHON_VERSION* \
    && ln -fs /usr/local/bin/python3 /usr/bin/python3 \
    && ln -fs /usr/local/bin/python3 /usr/bin/python

# Now install Poetry after Python is properly set up
RUN mkdir -p /opt/poetry \
    && curl -sSL https://install.python-poetry.org | POETRY_HOME=/opt/poetry python3 \
    && ln -s /opt/poetry/bin/poetry /usr/local/bin/poetry \
    && poetry config virtualenvs.create false

# Cleanup
RUN apt-get autoremove -y && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /var/cache/apt/* /usr/local/src/* /tmp/*

# Setup user
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m -s /bin/bash $USERNAME \
    && mkdir -p /workspace /home/$USERNAME/.cache /home/$USERNAME/.local \
    && chown -R $USERNAME:$USERNAME /workspace /home/$USERNAME /opt/poetry

WORKDIR /workspace
COPY --chown=$USERNAME:$USERNAME ./pyproject.toml ./poetry.lock* /workspace/
USER $USERNAME
RUN poetry install --no-root

CMD ["/bin/bash"]