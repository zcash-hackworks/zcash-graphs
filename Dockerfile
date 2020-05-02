FROM ubuntu:18.04

RUN export APT_FRONTEND=noninteractive && \
    apt-get update && \
    apt-get upgrade && \
    apt-get install -y \
        python3 \
        python3-pip \
        libcurl4-openssl-dev \
        libssl-dev

RUN pip3 install slick-bitcoinrpc && pip3 install progressbar2

RUN mkdir -p /project/{src,output}
COPY . /project/src

ENTRYPOINT ["python3", "/project/src/4467-sprout-usage/grab_shielded_data.py", "/project/output"]
