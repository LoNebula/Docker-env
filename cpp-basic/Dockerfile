FROM ubuntu:latest

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        gdb \
        cmake \
        git \
        && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY . /app

RUN cmake .
RUN make

CMD ["./myprogram"]