FROM rapidsai/rapidsai-core-dev:22.04-cuda11.5-devel-ubuntu20.04-py3.8
VOLUME /output

COPY build.sh build.sh
COPY cpp cpp
RUN /bin/bash -c '. /opt/conda/etc/profile.d/conda.sh && conda activate rapids && ./build.sh bench'
COPY run.sh run.sh

ENTRYPOINT ["./run.sh"]
