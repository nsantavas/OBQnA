FROM  nvidia/cuda:11.1-runtime
WORKDIR /app

ENV TZ=Europe/Amsterdam \
    LC_ALL=C.UTF-8 \
    LANG=C.UTF-8 \
    LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/nvidia/lib64 \
    PATH=${PATH}:/usr/local/nvidia/bin

RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

COPY ./requirements.txt ./requirements.txt

RUN apt-get update -y -qq && apt-get install -y --no-install-recommends \
    python3.8 python3-dev python3-setuptools python3-distutils python3-pip libopenblas-dev python3-numpy swig git \
    && export TMPDIR="/var/tmp" \
    && pip3 install -U pip  --no-cache-dir && pip3 install -r requirements.txt --no-cache-dir \
    && apt-get autoremove -y --purge && apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
COPY /app ./

EXPOSE 8080
#CMD ["gunicorn", "-w", "2", "-b", "0.0.0.0:8686", "-k", "uvicorn.workers.UvicornH11Worker", "--timeout", "120", "main:app"]
CMD ["gunicorn", "-b", "0.0.0.0:8686", "-k", "uvicorn.workers.UvicornH11Worker", "--timeout", "120", "main:app"]
