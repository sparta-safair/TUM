FROM ubuntu:16.04

RUN apt-get update && apt-get -y install wget gzip cmake python3-pip

ADD setup/setup-metis.sh /agad/setup-metis.sh

WORKDIR /agad
RUN ./setup-metis.sh
RUN rm setup-metis.sh

ADD setup/setup-reqs.sh /agad/setup-reqs.sh
ADD setup/requirements.txt /agad/requirements.txt

RUN ./setup-reqs.sh
RUN rm setup-reqs.sh requirements.txt

ADD . /agad
RUN setup/setup-data.sh

ENV PATH_REPO_AGAD /agad
ENV METIS_DLL /root/bin/metis-5.1.0/build/Linux-x86_64/libmetis/libmetis.so
ENV LC_ALL C.UTF-8
ENV LANG C.UTF-8
