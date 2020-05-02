FROM ubuntu:18.04

RUN export APT_FRONTEND=noninteractive && \
    apt-get update && \
    apt-get upgrade && \
    apt-get install -y \
        python \
        python-pip \
        libcurl4-openssl-dev \
        libssl-dev

RUN pip install slick-bitcoinrpc && pip install progressbar

RUN mkdir -p /project/{src,output}
COPY . /project/src

CMD python /project/src/4467-sprout-usage/grab_shielded_data.py
