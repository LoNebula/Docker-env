<!-- コンテナをビルド・起動します -->
docker-compose up --build

<!-- コンテナ表示一覧 -->
<!-- 実行中 -->
docker ps
<!-- 停止中も含めた全てのコンテナ -->
docker ps -a

<!-- イメージの一覧表示 -->
docker images

<!-- コンテナを停止する -->
docker-compose down

<!-- 停止中の全てのコンテナを削除 -->
docker container prune

<!-- 特定のコンテナを削除 -->
docker rm <コンテナ名>

<!-- 停止中だけでなく、実行中も含めた全てのコンテナ削除 -->
docker stop $(docker ps -aq) && docker rm $(docker ps -aq)

<!-- イメージIDを指定して削除 -->
docker rmi <イメージID>

<!-- 使われていないイメージを削除するには -->
docker image prune

<!-- 全てのイメージ削除 -->
docker rmi $(docker images -q)

<!-- 既存のコンテナとイメージを全て削除 -->
docker-compose down --rmi all

<!-- まとめて一括削除 -->
docker stop $(docker ps -aq)            # 全てのコンテナを停止
docker rm $(docker ps -aq)             # 全てのコンテナを削除
docker rmi $(docker images -q)         # 全てのイメージを削除

<!-- コンテナ内に入る -->
docker exec -it <コンテナ名> bash

<!-- コンテナをバックグラウンドで実行 -->
<!-- ubuntu-containerはオプションでコンテナに名前をつけている
my-appはビルドしたDockerイメージの名前--> -->
docker run -d --name ubuntu-container my-app

<!-- コンテナのログを確認 -->
docker logs my-python-container
