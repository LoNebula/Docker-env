# %%
import torch
import json
import os
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
import optuna
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import joblib
import sys
import io
from torchsummary import summary
from torchviz import make_dot
import random
from colorama import Fore, Style
import matplotlib.pyplot as plt
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed

# %%
# CPUスレッド数を設定
torch.set_num_threads(3)

"""parameter"""
# ランダムシード
random_seed = 42
# バリデーションの割合
val_split = 0.2
# inputdata数, outputdata数
input_dim = 3
output_dim = 2
# early stopping
patience = 10000  # 検証損失が改善しない許容エポック数
delta = 1e-4   # 検証損失の改善として認める最小値
# トライアル数
n_trials = 5
# 推論データ
test_x = 3 # x軸
test_y = 3 # y軸
test_z = 3 # z軸
# 学習率
lr_min = 1e-6
lr_max = 1e-1
# バッチサイズ
batch_size_min = 4
batch_size_max = 64
# エポック数
epochs_min = 10
epochs_max = 10000
# モデルの層数
n_layer_min = 1
n_layer_max = 20
# 各層のユニット数
model_unit_min = 4
model_unit_max = 8192
# ドロップアウト
drop_out = 0.4

# %%
def normalize_json_data(input_path, output_path, scaler_dir="../data/"):
    try:
        # JSONファイルを読み込み
        with open(input_path, "r") as file:
            data = json.load(file)
        
        # 必要なキーが存在するか確認
        if "inputs" not in data or "outputs" not in data:
            print("Error: 'inputs' or 'outputs' key not found in the JSON file.")
            return
        
        # 入力データと出力データをNumPy配列に変換
        input_data = np.array(data["inputs"])
        output_data = np.array(data["outputs"])
    
    except FileNotFoundError:
        print(f"Error: {input_path} not found.")
        return
    except json.JSONDecodeError:
        print(f"Error: Failed to parse JSON file: {input_path}.")
        return
    except Exception as e:
        print(f"Unexpected error while reading JSON file: {e}")
        return

    # スケーラーをインスタンス化
    input_scaler = MinMaxScaler()
    output_scaler = MinMaxScaler()

    try:
        # 入力と出力を正規化
        inputs_normalized = input_scaler.fit_transform(input_data)
        outputs_normalized = output_scaler.fit_transform(output_data)

        # 正規化データを辞書にまとめる
        normalized_data = {
            "inputs": inputs_normalized.tolist(),
            "outputs": outputs_normalized.tolist()
        }

        # 正規化データをファイルに保存
        with open(output_path, "w") as output_file:
            json.dump(normalized_data, output_file)
        
        # スケーラーを保存するディレクトリを作成（存在しない場合）
        os.makedirs(scaler_dir, exist_ok=True)
        
        # スケーラーを保存
        input_scaler_path = os.path.join(scaler_dir, "input_scaler.pkl")
        output_scaler_path = os.path.join(scaler_dir, "output_scaler.pkl")
        with open(input_scaler_path, "wb") as input_scaler_file:
            joblib.dump(input_scaler, input_scaler_file)
        with open(output_scaler_path, "wb") as output_scaler_file:
            joblib.dump(output_scaler, output_scaler_file)

        print("Normalization and scaler saving completed successfully.")
    
    except Exception as e:
        print(f"Error during normalization or saving process: {e}")


# %%
class CustomDataset(Dataset):
    def __init__(self, json_path):
        with open(json_path, 'r') as f:
            data = json.load(f)
        self.inputs = data["inputs"]
        self.outputs = data["outputs"]

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        x = torch.tensor(self.inputs[idx], dtype=torch.float32)
        y = torch.tensor(self.outputs[idx], dtype=torch.float32)
        return x, y

# %%
def get_dataloaders(json_path, batch_size=16, val_split=val_split, random_seed=random_seed):
    dataset = CustomDataset(json_path)
    
    # 乱数シードを設定
    torch.manual_seed(random_seed)
    random.seed(random_seed)
    
    # 訓練データと検証データのサイズ計算
    dataset_size = len(dataset)
    val_size = int(dataset_size * val_split)
    train_size = dataset_size - val_size
    
    # データセットを分割
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # DataLoaderの作成
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader

# %%
class NetworkModel(nn.Module):
    def __init__(self, input_dim, output_dim, trial, n_layers_min=n_layer_min, n_layers_max=n_layer_max, model_unit_min=model_unit_min, model_unit_max=model_unit_max, drop_out=drop_out):
        super(NetworkModel, self).__init__()

        # 層数をOptunaで指定
        n_layers = trial.suggest_int("n_layers", n_layers_min, n_layers_max)

        # 各層のユニット数をOptunaで指定
        layers = []
        in_features = input_dim
        for i in range(n_layers):
            out_features = trial.suggest_int(f"n_units_l{i}", model_unit_min, model_unit_max)  # 各層のユニット数
            layers.append(nn.Linear(in_features, out_features))
            layers.append(nn.ReLU())
            if trial.suggest_float(f"dropout_l{i}", 0.0, drop_out) > 0.0:
                layers.append(nn.Dropout(trial.suggest_float(f"dropout_l{i}", 0.0, drop_out)))
            in_features = out_features
        
        # 出力層
        layers.append(nn.Linear(in_features, output_dim))
        
        # nn.Sequentialにまとめる
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

# %%
# トライアル終了後のコールバックで保存
def save_trial(trial, model, save_dir="../results/model"):
    trial_params = trial.params
    trial_number = trial.number + 1
    
    # 保存するディレクトリを作成
    os.makedirs(save_dir, exist_ok=True)

    # ハイパーパラメータをJSONファイルに保存
    with open(f"{save_dir}/trial_{trial_number}_params.json", "w") as f:
        json.dump(trial_params, f, indent=4)

    # モデルの状態を保存
    torch.save(model.state_dict(), f"{save_dir}/trial_{trial_number}_model.pth")

# %%
def clean_and_prepare_directories():
    # クリーンアップ対象のディレクトリを指定
    directories_to_clean = [
        "../data/summary",
        "../images/model_graph",
        "../images/graph_training_results",
        "../images/graph_val_results",
        "../results"
    ]
    
    # 各ディレクトリ内の既存ファイルを削除
    for dir_path in directories_to_clean:
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path)  # ディレクトリとその中身を削除
        os.makedirs(dir_path, exist_ok=True)  # 再作成

    print("Directories cleaned and prepared.")

# トレーニング開始前にディレクトリをクリーンアップ
clean_and_prepare_directories()

# %%
# モデルを作成する関数
def create_model(trial, input_dim, output_dim):
    return NetworkModel(input_dim, output_dim, trial)

# %%
def train_model(trial, input_dim=input_dim, output_dim=output_dim, patience=patience, delta=delta, lr_min=lr_min, lr_max=lr_max, batch_size_min=batch_size_min, batch_size_max=batch_size_max, epochs_min=epochs_min, epochs_max=epochs_max):
    # モデルの作成
    input_dim = input_dim
    output_dim = output_dim
    model = create_model(trial, input_dim, output_dim)
    
    # モデルを trial.user_attrs に保存
    trial.set_user_attr("model", model)
    
        # トライアルごとに保存
    trial_number = trial.number + 1  # トライアル番号
    
    # Early Stoppingのパラメータ
    patience = patience  # 検証損失が改善しない許容エポック数
    delta = delta   # 検証損失の改善として認める最小値
    early_stop_counter = 0  # 改善しないエポック数をカウント
    
    # モデルサマリーの保存
    os.makedirs("../data/summary", exist_ok=True)
    stdout_backup = sys.stdout  # 標準出力のバックアップ
    sys.stdout = io.StringIO()  # 標準出力をキャプチャ
    summary(model, (1, input_dim))  # モデルのサマリーを取得 (バッチサイズ1の例)
    summary_str = sys.stdout.getvalue()  # サマリー内容を取得
    sys.stdout = stdout_backup  # 標準出力を元に戻す
    with open(f"../data/summary/model_summary_trial_{trial_number}.txt", "w") as f:
        f.write(summary_str)

    # 計算グラフを保存
    os.makedirs("../images/model_graph/", exist_ok=True)
    x = torch.randn(1, input_dim)  # ダミー入力データを作成
    y = model(x)  # ダミー入力を通して出力を取得
    dot = make_dot(y, params=dict(model.named_parameters()))  # 計算グラフを作成
    dot.format = "png"  # PNG形式で保存
    dot.render(f"../images/model_graph/model_graph_trial_{trial_number}")  # ファイル名にトライアル番号を付与
    
    
    # Optunaのトライアルからハイパーパラメータを取得
    lr = trial.suggest_float("lr", lr_min, lr_max)
    batch_size = trial.suggest_int("batch_size", batch_size_min, batch_size_max)
    epochs = trial.suggest_int("epochs", epochs_min, epochs_max)
    optimizer_name = trial.suggest_categorical(
    "optimizer", ["Adam", "SGD", "RMSprop", "AdamW", "Adagrad", "Adadelta", "Adamax"]
)
    loss_function_name = trial.suggest_categorical(
    "loss_function", ["MSELoss", "L1Loss", "SmoothL1Loss", "HuberLoss", "BCEWithLogitsLoss", "CrossEntropyLoss"]
)
    
    # Optimizer を選択
    if optimizer_name == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=lr)
    elif optimizer_name == "SGD":
        momentum = trial.suggest_float("momentum", 0.0, 0.9)  # SGD の場合のみ momentum を提案
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    elif optimizer_name == "RMSprop":
        optimizer = optim.RMSprop(model.parameters(), lr=lr)
    elif optimizer_name == "AdamW":
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2)  # AdamW用のweight decay
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == "Adagrad":
        optimizer = optim.Adagrad(model.parameters(), lr=lr)
    elif optimizer_name == "Adadelta":
        optimizer = optim.Adadelta(model.parameters(), lr=lr)
    elif optimizer_name == "Adamax":
        optimizer = optim.Adamax(model.parameters(), lr=lr)
            
    # 損失関数を選択
    if loss_function_name == "MSELoss":
        criterion = nn.MSELoss()
    elif loss_function_name == "L1Loss":
        criterion = nn.L1Loss()
    elif loss_function_name == "SmoothL1Loss":
        beta = trial.suggest_float("beta", 0.1, 1.0)  # SmoothL1Loss 用の beta を提案
        criterion = nn.SmoothL1Loss(beta=beta)
    elif loss_function_name == "HuberLoss":
        delta = trial.suggest_float("delta", 1.0, 10.0)  # HuberLoss用のdelta
        criterion = nn.HuberLoss(delta=delta)
    elif loss_function_name == "BCEWithLogitsLoss":
        criterion = nn.BCEWithLogitsLoss()
    elif loss_function_name == "CrossEntropyLoss":
        criterion = nn.CrossEntropyLoss()
    
    # データローダーの取得
    train_loader, val_loader = get_dataloaders('../data/normalized_data.json', batch_size=batch_size)
    
    train_losses = []  # 訓練損失を記録
    val_losses = [] #検証損失の記録
    val_accuracies = [] #バリデーション精度の記録
    
    best_loss = float("inf")
    best_model = None
    
    # トレーニングの進行状況をプログレスバーで表示
    progress_bar = tqdm(
        range(epochs),
        desc=f"Trial {trial_number} Progress",  # トライアル番号付きの進行状況
        unit="epoch",
        bar_format="{l_bar}{bar:40}| {n_fmt}/{total_fmt} epochs [{elapsed}<{remaining}] - {postfix}"
        )
    
    # エポックごとの訓練ループ
    for epoch in progress_bar:
        # 訓練フェーズ
        model.train()  # モデルを訓練モードに設定
        total_train_loss = 0  # 訓練損失の合計

        for x, y in train_loader:
            optimizer.zero_grad()  # 勾配をゼロにリセット
            y_pred = model(x)  # モデルで予測
            loss = criterion(y_pred, y)  # 損失を計算
            loss.backward()  # 逆伝播
            optimizer.step()  # パラメータを更新
            total_train_loss += loss.item()  # 訓練損失を累積

        avg_train_loss = total_train_loss / len(train_loader)  # 平均訓練損失
        train_losses.append(avg_train_loss)  # 訓練損失を記録

        # 検証フェーズ
        model.eval()  # モデルを評価モードに設定
        total_val_loss = 0  # 検証損失の合計
        correct_predictions = 0  # 正しい予測の数
        total_samples = 0  # サンプル数

        with torch.no_grad():  # 検証時は勾配計算を行わない
            for x, y in val_loader:
                y_pred = model(x)  # モデルで予測
                loss = criterion(y_pred, y)  # 損失を計算
                total_val_loss += loss.item()  # 検証損失を累積

                # 精度を計算
                _, predicted = torch.max(y_pred, 1)  # 予測結果
                _, labels = torch.max(y, 1)  # 正解ラベル
                correct_predictions += (predicted == labels).sum().item()  # 正しい予測の数をカウント
                total_samples += labels.size(0)  # サンプル数をカウント

        avg_val_loss = total_val_loss / len(val_loader)  # 平均検証損失
        val_losses.append(avg_val_loss)  # 検証損失を記録

        accuracy = 100 * correct_predictions / total_samples  # 精度の計算
        val_accuracies.append(accuracy)  # 精度を記録

        # 最良モデルを保存
        if avg_val_loss < best_loss - delta:
            best_loss = avg_val_loss
            best_model = model.state_dict()
            early_stop_counter = 0  # 改善があればカウンターをリセット

        else:
            early_stop_counter += 1  # 改善しない場合にカウンターを増加
            
         # プログレスバーに詳細な情報を表示
        progress_bar.set_postfix({
            "TrainLoss": f"{avg_train_loss:.4f}",
            "ValLoss": f"{avg_val_loss:.4f}",
            "ValAcc": f"{accuracy:.2f}%",
            "ES": f"{early_stop_counter}/{patience}",  # Early stoppingのカウントと許容値
            "LR": f"{lr:.2e}",  # 学習率を科学記号形式で表示
            "Batch": batch_size,
            "Opt": optimizer.__class__.__name__,
            "LossFn": loss_function_name
        })

        # Optunaの進捗を報告
        trial.report(avg_val_loss, epoch)
        
        # Early Stoppingの判定
        if early_stop_counter >= patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

        # プルーニングが必要ならば中断
        """if trial.should_prune():
            raise optuna.exceptions.TrialPruned()"""
        
        
        
    # トライアルごとに学習曲線とバリデーション精度をプロットして保存
    os.makedirs('../images/graph_training_results', exist_ok=True)  # images/graphディレクトリがない場合は作成
    os.makedirs('../images/graph_val_results', exist_ok=True)

    # 学習曲線をプロット
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title(f"Training and Validation Loss (Trial {trial_number})")
    plt.legend()
    plt.grid(True)
    plt.savefig(f'../images/graph_training_results/training_validation_loss_trial_{trial_number}.png')  # トライアルごとに保存
    plt.close()

    # バリデーション精度のグラフをプロット
    plt.figure(figsize=(10, 6))
    plt.plot(val_accuracies, label="Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy (%)")
    plt.title(f"Validation Accuracy (Trial {trial_number})")
    plt.legend()
    plt.grid(True)
    plt.savefig(f'../images/graph_val_results/validation_accuracy_trial_{trial_number}.png')  # トライアルごとに保存
    plt.close()


    if not os.path.exists('../results'):
        os.makedirs('../results')
        
    # トライアルごとに最良モデルを保存
    torch.save(best_model, f'../results/best_model_trial_{trial_number}.pth')  # トライアルごとに保存

    # 最良モデル（全トライアル）を保存
    if best_model is not None:
        # 最初のトライアルであれば最良モデルを保存、次のトライアルで更新
        if trial.number == 0 or avg_val_loss < best_loss:
            torch.save(best_model, '../results/best_model.pth')

    return best_loss # 最良の検証損失を返す

# %%
def perform_bayesian_optimization(n_trials):
    """Optunaによるベイズ最適化の実行"""
    study = optuna.create_study(direction='minimize')
    
    # プログレスバーの設定
    progress_bar = tqdm(
        total=n_trials,
        desc=f"{Fore.MAGENTA}Bayesian Optimization Progress{Style.RESET_ALL}",
        unit="trial",
        bar_format="{l_bar}{bar:30}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] - {desc}"
    )

    def callback(study, trial):
        # プログレスバーの更新
        progress_bar.set_description(
            f"{Fore.BLUE}Trial {trial.number+1}{Style.RESET_ALL} | Best Loss: {Fore.RED}{study.best_value:.6f}{Style.RESET_ALL}"
        )
        progress_bar.update(1)

        # ベストハイパーパラメータと損失をリアルタイム表示
        print(f"\n{Fore.YELLOW}Current Best Hyperparameters:{Style.RESET_ALL} {study.best_params}")
        print(f"{Fore.YELLOW}Current Best Loss:{Style.RESET_ALL} {study.best_value:.6f}")

    # ベイズ最適化の実行
    study.optimize(
        lambda trial: train_model(trial),  # 損失値を返す関数
        n_trials=n_trials,
        callbacks=[callback, lambda study, trial: save_trial(trial, trial.user_attrs["model"])]
    )

    # プログレスバーを閉じる
    progress_bar.close()

    # 最終結果の表示
    best_trial = study.best_trial
    print(f"{Fore.GREEN}Best Trial Number:{Style.RESET_ALL} {best_trial.number+1}")
    print(f"{Fore.GREEN}Best Hyperparameters:{Style.RESET_ALL} {best_trial.params}")
    print(f"{Fore.GREEN}Best Loss:{Style.RESET_ALL} {best_trial.value}")

    return best_trial.params, best_trial.number + 1


# %%
# JSONデータの正規化
normalize_json_data('../data/data.json', '../data/normalized_data.json')

# Optunaによるベイズ最適化の実行
best_params, best_number = perform_bayesian_optimization(n_trials=n_trials)
print("Optimization completed. Best parameters:", best_params)

# %%
def load_model_from_trial(input_dim=input_dim, output_dim=output_dim, params_path=f"../results/model/trial_{best_number}_params.json", model_path=f"../results/model/trial_{best_number}_model.pth"):
    # ハイパーパラメータを読み込む
    with open(params_path, "r") as f:
        params = json.load(f)
    
    # ダミーのトライアルオブジェクトを作成
    class DummyTrial:
        def __init__(self, params):
            self.params = params
        
        def suggest_int(self, name, low, high):
            return self.params[name]
        
        def suggest_float(self, name, low, high):
            return self.params[name]
    
    dummy_trial = DummyTrial(params)
    
    # モデルを構築
    model = NetworkModel(input_dim, output_dim, dummy_trial)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    input_scaler = joblib.load('../data/input_scaler.pkl')
    output_scaler = joblib.load('../data/output_scaler.pkl')
    
    return model, input_scaler, output_scaler

model, input_scaler, output_scaler = load_model_from_trial()

# %%
 # 推論用関数
def generate_output(input_data, model=model, input_scaler=input_scaler, output_scaler=output_scaler):
    input_data_scaled = input_scaler.transform([input_data])
    input_tensor = torch.tensor(input_data_scaled, dtype=torch.float32)

    with torch.no_grad():
        output_scaled = model(input_tensor)
        output = output_scaler.inverse_transform(output_scaled.numpy())

    return output[0]

# %%
# 推論
test_input = [test_x, test_y, test_z]
output = generate_output(test_input)
print(f"Input data: {test_input}")
print(f"Generated Output: {output}")

# %%

# 最大値を初期化
max_input = [0, 0, 0]
max_output = [-1e+1000000, -1e+1000000]

# 出力を計算する関数
def process_point(i, j, k, model, input_scaler, output_scaler):
    inference_input = [i, j, k]
    out_put = generate_output(inference_input, model, input_scaler, output_scaler)
    return out_put[1], [inference_input[0], inference_input[1], inference_input[2]], out_put

# tqdmで進行状況を表示
with ThreadPoolExecutor() as executor:
    futures = []
    for i in tqdm(range(100), desc="データ収集"):
        for j in range(100):
            for k in range(100):
                # 各スレッドに処理を割り当て
                futures.append(executor.submit(process_point, i, j, k, model, input_scaler, output_scaler))

    # 結果を処理
    for future in tqdm(as_completed(futures), desc="結果処理中", total=len(futures)):
        out_put_value, input_value, output_value = future.result()
        if max_output[1] < out_put_value:
            max_output = output_value
            max_input = input_value

print("Max Input:", max_input)
print("Max Output:", max_output)



