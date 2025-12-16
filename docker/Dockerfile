FROM supervisely/is-net:1.0.1
ENV DEBIAN_FRONTEND=noninteractive

RUN apt update && sudo apt upgrade -y
RUN python3 -m pip install supervisely==6.73.486

LABEL python_sdk_version="6.73.486"
