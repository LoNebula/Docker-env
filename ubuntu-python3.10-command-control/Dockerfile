FROM ubuntu:22.04

# Python3.10と必要なツールをインストール
RUN apt-get update && apt-get install -y \
    software-properties-common \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get install -y python3.10 python3.10-venv python3-pip \
    && apt-get clean

# 作業ディレクトリを設定
WORKDIR /usr/src

# Pythonの依存関係をインストール
COPY requirements.txt ./
RUN pip3 install --no-cache-dir -r requirements.txt

# アプリケーションコードをコピー
COPY src ./src

# コンテナ起動時にbashシェルを実行
ENTRYPOINT ["/bin/bash"]

# デフォルトのコマンドを指定 (例: 任意のコマンドを指定するため)
CMD ["-c", "while :; do sleep 1000; done"]
