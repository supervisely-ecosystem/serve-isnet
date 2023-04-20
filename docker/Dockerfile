FROM supervisely/base-py-sdk:6.69.78
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update

RUN pip3 install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu116
RUN pip3 install gdown

RUN python3 -m pip install supervisely==6.69.78

RUN mkdir -p weights
COPY isnet.pth /weights/