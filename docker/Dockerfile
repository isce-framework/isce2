FROM ubuntu:20.04 as builder

# Set an encoding to make things work smoothly.
ENV LANG en_US.UTF-8
ENV TZ US/Pacific
ARG DEBIAN_FRONTEND=noninteractive

RUN set -ex \
 && apt-get update \
 && apt-get install -y \
    cmake \
    cython3 \
    git \
    libfftw3-dev \
    libgdal-dev \
    libhdf4-alt-dev \
    libhdf5-dev \
    libopencv-dev \
    ninja-build \
    python3-gdal \
    python3-h5py \
    python3-numpy \
    python3-scipy \
 && echo done

# copy repo
COPY . /opt/isce2/src/isce2

# build ISCE
RUN set -ex \
 && cd /opt/isce2/src/isce2 \
 && mkdir build && cd build \
 && cmake .. \
        -DPYTHON_MODULE_DIR="$(python3 -c 'import site; print(site.getsitepackages()[-1])')" \
        -DCMAKE_INSTALL_PREFIX=install \
 && make -j8 install \
 && cpack -G DEB \
 && cp isce*.deb /tmp/

FROM ubuntu:20.04

# Set an encoding to make things work smoothly.
ENV LANG en_US.UTF-8
ENV TZ US/Pacific
ARG DEBIAN_FRONTEND=noninteractive

RUN set -ex \
 && apt-get update \
 && apt-get install -y \
    libfftw3-3 \
    libgdal26 \
    libhdf4-0 \
    libhdf5-103 \
    libopencv-core4.2 \
    libopencv-highgui4.2 \
    libopencv-imgproc4.2 \
    python3-gdal \
    python3-h5py \
    python3-numpy \
    python3-scipy \
 && echo done

# install ISCE from DEB
COPY --from=builder /tmp/isce*.deb /tmp/isce2.deb

RUN dpkg -i /tmp/isce2.deb
