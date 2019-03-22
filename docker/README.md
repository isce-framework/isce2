# ISCE in docker

## Build

1. Clone repo:
   ```
   git clone https://github.com/isce-framework/isce2.git
   ```
1. Change directory:
   ```
   cd isce2
   ```
1. Build image:
   ```
   docker build --rm --force-rm -t hysds/isce_giant -f docker/Dockerfile .
   ```
