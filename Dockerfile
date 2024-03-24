FROM nvidia/cuda:12.2.0-runtime-ubuntu20.04 as builder


# Update and install all required packages in a single RUN command to reduce layers
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    python3.9 \
    python3-pip \
    && ln -s /usr/bin/python3.9 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*


RUN pip3 install --upgrade pip

COPY ./requirements.txt .
RUN pip3 install -r requirements.txt 


COPY src ./opt/src

COPY ./entry_point.sh /opt/
RUN chmod +x /opt/entry_point.sh

WORKDIR /opt/src

ENV MPLCONFIGDIR=/tmp/matplotlib
ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV PATH="/opt/app:${PATH}"

RUN mkdir -p /opt/src/lightning_logs && chmod -R 777 /opt/src/lightning_logs


# set non-root user
USER 1000
# set entrypoint
ENTRYPOINT ["/opt/entry_point.sh"]