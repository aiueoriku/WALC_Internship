import itertools
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import graphviz
import mglearn

from scipy.cluster.hierarchy import dendrogram, fcluster, ward
from sklearn.cluster import DBSCAN, KMeans
from sklearn.datasets import load_iris
from sklearn.decomposition import NMF, PCA
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import adjusted_rand_score
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler, Normalizer, RobustScaler, StandardScaler
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier, export_graphviz


class AnalyzeIris:
    """Irisデータセットを分析するクラス"""

    def __init__(self,
                 random_state: int = 0,
                 n_neighbors: int = 4,
                 max_depth: int = 4,
                 n_splits: int = 5,
                 shuffle: bool = True):
        """コンストラクタ

        Args:
            random_state (int): 乱数シード
            n_neighbors (int): KNeighborsClassifierの近傍数
            max_depth (int): DecisionTreeClassifierの最大深度
            n_splits (int): KFoldの分割数
            shuffle (bool): KFoldでデータをシャッフルするかどうか
        """
        
        # Irisデータセットの読み込み
        iris = load_iris()
        self.X = iris.data
        self.y = iris.target
        self.feature_names = iris.feature_names
        self.target_names = iris.target_names

        # DataFrameの作成
        self.df_data = pd.DataFrame(self.X, columns=self.feature_names)
        self.df_data_with_label = self.df_data.copy()
        self.df_data_with_label["label"] = self.y
        
        # スコア，学習済みモデルを保持
        self.scores = {}
        self.trained_models = {}
        
        # ハイパーパラメータの保存
        self.random_state = random_state
        self.n_neighbors = n_neighbors
        self.max_depth = max_depth
        self.n_splits = n_splits
        self.shuffle = shuffle
        
        self.models = self._init_models()
        
        self.scalers = [
            MinMaxScaler(),
            StandardScaler(),
            RobustScaler(),
            Normalizer()
        ]

    def _init_models(self):
        """使用するモデルを初期化する"""
        return [
            ("LogisticRegression", LogisticRegression(random_state=self.random_state)),
            ("LinearSVC", LinearSVC(random_state=self.random_state)),
            ("SVC", SVC(random_state=self.random_state)),
            ("DecisionTreeClassifier", DecisionTreeClassifier(random_state=self.random_state, max_depth=self.max_depth)),
            ("KNeighborsClassifier", KNeighborsClassifier(n_neighbors=self.n_neighbors)),
            ("LinearRegression", LinearRegression()),
            ("RandomForestClassifier", RandomForestClassifier(random_state=self.random_state)),
            ("GradientBoostingClassifier", GradientBoostingClassifier(random_state=self.random_state)),
            ("MLPClassifier", MLPClassifier(random_state=self.random_state))
        ]

    def get(self):
        """ラベル付のデータフレームを返す

        Returns:
            pd.DataFrame: ラベルを含むデータフレーム
        """
        return self.df_data_with_label

    def get_correlation(self):
        """データフレームの相関行列を計算

        Returns:
            pd.DataFrame: データフレームの相関行列

        """
        return self.df_data.corr()

    def pair_plot(self, diag_kind: str = "hist") -> sns.PairGrid:
        """ペアプロットを描画する

        Args:
            diag_kind (str): 対角プロットの種類（例："hist"）

        Returns:
            sns.PairGrid: 作成したペアプロットオブジェクト
        """
        df_data_with_label_str = self.df_data_with_label.copy()
        df_data_with_label_str["label"][df_data_with_label_str["label"] == 0] = "setosa"
        df_data_with_label_str["label"][df_data_with_label_str["label"] == 1] = "versicolor"
        df_data_with_label_str["label"][df_data_with_label_str["label"] == 2] = "virginica"
        return sns.pairplot(df_data_with_label_str, hue="label", diag_kind=diag_kind)

    def all_supervised(self, n_neighbors=None) -> None:
        """複数の教師あり学習モデルを評価する

        KFoldを用いて、各モデルを学習・評価し、スコアを保存する。
        """
        # n_neighborsが指定されていれば上書きしてモデルを再生成
        if n_neighbors is not None:
            self.n_neighbors = n_neighbors
            self.models = self._init_models()
            
        kf = KFold(
            n_splits=self.n_splits,
            shuffle=self.shuffle,
            random_state=self.random_state
        )

        # スコアと学習済みモデルをリセット
        self.scores = {}
        self.trained_models = {}

        for model_name, model in self.models:
            print(f"=== {model_name} ===")

            self.scores[model_name] = []

            for train_index, test_index in kf.split(self.X):
                X_train, X_test = self.X[train_index], self.X[test_index]
                y_train, y_test = self.y[train_index], self.y[test_index]

                model.fit(X_train, y_train)

                train_score = model.score(X_train, y_train)
                test_score = model.score(X_test, y_test)

                self.scores[model_name].append(test_score)
                self.trained_models[model_name] = model

                print(f"test score: {test_score:.3f}, train score: {train_score:.3f}")

    def get_supervised(self) -> pd.DataFrame:
        """教師あり学習モデルの評価を行いpandas.DataFrameで返す

        Returns:
            pd.DataFrame: 教師あり学習モデルの評価結果
        """

        if not self.scores:
            # 評価が行われていない場合、現在の引数で all_supervised を実行
            self.all_supervised(n_splits=self.n_splits, shuffle=self.shuffle, random_state=self.random_state)
        else:
            # 既にスコアがある場合、引数が異なるかを確認し、異なれば再実行
            if self.n_splits != self.n_splits or self.shuffle != self.shuffle or self.random_state != self.random_state:
                print("引数が変更されたため、再評価を行います。")
                self.all_supervised(n_splits=self.n_splits, shuffle=self.shuffle, random_state=self.random_state)

        df_results = pd.DataFrame(self.scores)
        return df_results

    def best_supervised(self):
        """教師あり学習モデルの中で最も性能が良いモデルを返す

        Returns:
            str: 最も性能が良いモデルの名前
            float: 最も性能が良いモデルの平均スコア
        """
        df_results = self.get_supervised()
        mean = df_results.mean()  # 各列の平均値を計算
        best_method = mean.idxmax()  # 最大値を持つ列の名前を取得
        best_score = mean.max()  # 最大値を取得
        return best_method, best_score

    def plot_feature_importances_all(self):
        """全てのモデルの特徴量の重要度をプロットする
        """
        for _, model in self.trained_models.items():
            if hasattr(model, "feature_importances_"):
                # モデルが提供する feature_importances_ に基づいて特徴量数を取得
                n_features = len(model.feature_importances_)
                feature_names = self.df_data.columns[:n_features]
                
                # プロット
                plt.barh(range(n_features), model.feature_importances_, align="center")
                plt.yticks(np.arange(n_features), feature_names)
                plt.xlabel(f"Feature importance: {model.__class__.__name__}")
                plt.title(f"Feature Importance of {model.__class__.__name__}")
                plt.show()

    def visualize_decision_tree(self):
        """決定木の可視化
        
        Returns:
            graphviz.Source: 決定木の可視化結果
        """
        for model_name, model in self.trained_models.items():
            if model_name == "DecisionTreeClassifier":
                export_graphviz(
                    model,
                    out_file=f"{model_name}.dot",
                    feature_names=self.df_data_with_label.columns[:-1],
                    class_names=["setosa", "versicolor", "virginica"],
                    filled=True,
                    rounded=True,
                )
                with open(f"{model_name}.dot") as f:
                    dot_graph = f.read()
                graph = graphviz.Source(dot_graph)
                return graph

    def plot_scaled_data(self) -> None:
        """各スケーリング手法の効果をLinearSVCの結果とともにプロットする"""

        kf = KFold(n_splits=self.n_splits, shuffle=self.shuffle, random_state=self.random_state)
        
        num_features = self.X.shape[1]
        feature_names = self.df_data.columns[:num_features]
        plot_combinations = list(itertools.combinations(range(num_features), 2))

        for train_index, test_index in kf.split(self.X):
            X_train, X_test = self.X[train_index], self.X[test_index]
            y_train, y_test = self.y[train_index], self.y[test_index]

            # スケーラーを適用しない場合の結果を出力
            model = LinearSVC(random_state=self.random_state, max_iter=10000)
            model.fit(X_train, y_train)

            train_score = model.score(X_train, y_train)
            test_score = model.score(X_test, y_test)
            print(f"Original: test score: {test_score:.3f}   train score: {train_score:.3f}")

            # プロット用の設定
            fig, ax = plt.subplots(6, 5, figsize=(13, 15))
            ax = ax.ravel()  # 1次元配列に変換

            # オリジナルデータの散布図
            for i, (x_idx, y_idx) in enumerate(plot_combinations):
                ax[i * 5].scatter(X_train[:, x_idx], X_train[:, y_idx], c='blue', marker='o', label='Train')
                ax[i * 5].scatter(X_test[:, x_idx], X_test[:, y_idx], c='red', marker='^', label='Test')
                ax[i * 5].set_title(f"Original")
                ax[i * 5].set_xlabel(feature_names[x_idx])
                ax[i * 5].set_ylabel(feature_names[y_idx])
                ax[i * 5].legend()

            # 各スケーリング手法ごとのプロット
            for j, scaler in enumerate(self.scalers):
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)

                model = LinearSVC(random_state=self.random_state, max_iter=10000)
                model.fit(X_train_scaled, y_train)

                train_score = model.score(X_train_scaled, y_train)
                test_score = model.score(X_test_scaled, y_test)

                print(f"{scaler.__class__.__name__}: test score: {test_score:.3f}   train score: {train_score:.3f}")

                # 各スケーリング手法での散布図
                for i, (x_idx, y_idx) in enumerate(plot_combinations):
                    idx = i * 5 + (j + 1)
                    ax[i * 5 + (j + 1)].scatter(X_train_scaled[:, x_idx], X_train_scaled[:, y_idx], c='blue', marker='o', label='Train')
                    ax[i * 5 + (j + 1)].scatter(X_test_scaled[:, x_idx], X_test_scaled[:, y_idx], c='red', marker='^', label='Test')
                    ax[i * 5 + (j + 1)].set_title(f"{scaler.__class__.__name__}")
                    ax[i * 5 + (j + 1)].legend()
                    ax[idx].set_xlabel(feature_names[x_idx])
                    ax[idx].set_ylabel(feature_names[y_idx])
                    ax[idx].legend()

            plt.tight_layout()
            plt.show()
            plt.close(fig)

            print("=" * 50)


    def _plot_dim_reduction(
        self,
        X_trans: np.ndarray,
        model,  # PCA or NMF (anything that has .components_)
        method_name: str,
        feature_names: list[str],
        y: np.ndarray,
        target_names: list[str],
        component_names: list[str] = None
    ):
        """
        PCAやNMFなどの次元削減結果を描画
        
        Args:
            X_trains(np.ndarray): 次元削減後のデータ
            model(PCA or NMF): 次元削減モデル
            method_name(str): 次元削減手法の名前
            feature_names(list[str]): 特徴量名
            y(np.ndarray): ターゲットラベル(0,1,2)
            target_names(list[str]): ターゲット名(["setosa", "versicolor", "virginica"])
            component_names(list[str]): 次元削減後の特徴量名(["Component 1", "Component 2", ...])
        """

        n_components = X_trans.shape[1]  # (サンプル数, 次元数)
        if component_names is None:
            # デフォルトラベル: ["Component 1", "Component 2", ...]
            component_names = [f"Component {i+1}" for i in range(n_components)]

        # --- 2次元散布図 ---
        plt.figure(figsize=(8, 6))
        mglearn.discrete_scatter(X_trans[:, 0], X_trans[:, 1], y)
        plt.legend(target_names, loc="best")
        plt.gca().set_aspect("equal")
        plt.xlabel(component_names[0])
        if n_components > 1:
            plt.ylabel(component_names[1])
        plt.title(method_name)
        # 軸の表示範囲を少し余裕を持たせる
        plt.ylim([X_trans[:, 1].min() - 0.5, X_trans[:, 1].max() + 0.5])
        plt.xlim([X_trans[:, 0].min() - 0.5, X_trans[:, 0].max() + 0.5])
        plt.show()

        # --- 成分行列のヒートマップ ---
        plt.figure(figsize=(8, 4))
        plt.matshow(model.components_, fignum=False, cmap='viridis')
        plt.colorbar()
        plt.xticks(range(len(feature_names)), feature_names, rotation=60, ha='left')
        plt.yticks(range(n_components), component_names)
        plt.xlabel("Feature")
        plt.ylabel(f"{method_name} components")
        plt.title(f"{method_name} Components", pad=20)  # タイトルが重ならないようにpadを調整
        plt.show()

    def plot_pca(self, n_components=2):
        """PCAによる次元削減を行い、2次元に削減したデータをプロットする

        Args:
            n_components (int, optional): 次元削減後の次元数. Default is 2.
            
        Returns:
            X_scaled_df (DataFrame): スケーリング後のデータ
            df_pca (DataFrame): PCAによる次元削減後のデータ
            pca_scaled (PCA): スケーリング後のPCAモデル
        """
        # スケーリング前のPCA
        pca = PCA(n_components=n_components)
        pca.fit(self.X)
        X_pca = pca.transform(self.X)
        
        self._plot_dim_reduction(
            X_trans=X_pca,
            model=pca,
            method_name="PCA (Original)",
            feature_names=self.feature_names,
            y=self.y,
            target_names=self.target_names,
            component_names=["First component", "Second component"] if n_components == 2 else None
        )

        # スケーリング後のPCA
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(self.X)
        
        pca_scaled = PCA(n_components=n_components)
        pca_scaled.fit(X_scaled)
        X_scaled_pca = pca_scaled.transform(X_scaled)
        
        self._plot_dim_reduction(
            X_trans=X_scaled_pca,
            model=pca_scaled,
            method_name="PCA (Scaled)",
            feature_names=self.feature_names,
            y=self.y,
            target_names=self.target_names,
            component_names=["First component", "Second component"] if n_components == 2 else None
        )
        
        X_scaled_df = pd.DataFrame(X_scaled, columns=self.feature_names)
        df_pca = pd.DataFrame(pca_scaled.components_, columns=self.feature_names)
        
        return X_scaled_df, df_pca, pca_scaled
    

    def plot_nmf(self, n_components=2):
            """NMFによる次元削減を行い、2次元に削減したデータをプロットする

            Args:
                n_components (int, optional): 次元削減後の次元数. Default is 2.
            
            Returns:
                X_scaled_df (DataFrame): スケーリング後のデータ
                df_nmf (DataFrame): nmfによる次元削減後のデータ
                nmf_scaled (nmf): スケーリング後のnmfモデル
            """

            # スケーリング前のNMF
            nmf = NMF(n_components=n_components, init='random', random_state=self.random_state)
            X_nmf = nmf.fit_transform(self.X)  # fitしてからtransformでも良いが fit_transformでOK

            # プロット
            self._plot_dim_reduction(
                X_trans=X_nmf,
                model=nmf,
                method_name="NMF (Original)",
                feature_names=self.feature_names,
                y=self.y,
                target_names=self.target_names,
                component_names=[f"Component {i+1}" for i in range(n_components)]
            )

            # スケーリング後のNMF
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(self.X)
            
            nmf_scaled = NMF(n_components=n_components, init='random', random_state=0)
            X_scaled_nmf = nmf_scaled.fit_transform(X_scaled - X_scaled.min())  # NMFは非負値のみ扱うため、最小値を調整

            # プロット
            self._plot_dim_reduction(
                X_trans=X_scaled_nmf,
                model=nmf_scaled,
                method_name="NMF (Scaled)",
                feature_names=self.feature_names,
                y=self.y,
                target_names=self.target_names,
                component_names=[f"Component {i+1}" for i in range(n_components)]
            )

            X_scaled_df = pd.DataFrame(X_scaled, columns=self.feature_names)
            df_nmf = pd.DataFrame(nmf_scaled.components_, columns=self.feature_names)
            
            return X_scaled_df, df_nmf, nmf_scaled
        
    def plot_tsne(self, random_state=0):
        """t-SNEによる次元削減を行い、2次元に削減したデータをプロットする

        Args:
            random_state (int, optional): 乱数のシード. Default is 0.
        """
        
        tsne = TSNE(random_state=random_state)
        X_tsne = tsne.fit_transform(self.X)
        
        # 各データ点をテキストとしてプロット
        for i in range(len(self.y)):
            plt.text(X_tsne[i, 0], X_tsne[i, 1], str(self.y[i]), 
                    fontsize=9, weight='bold', ha='center', va='center')

        # 軸のラベルとスケール調整
        plt.xlabel("t-SNE feature 0")
        plt.ylabel("t-SNE feature 1")
        plt.ylim([X_tsne[:, 1].min() - 0.5, X_tsne[:, 1].max() + 0.5])
        plt.xlim([X_tsne[:, 0].min() - 0.5, X_tsne[:, 0].max() + 0.5])
        
        # print(self.y)
    
    def plot_k_means(self, n_clusters=3):
        """KMeans法によるクラスタリングを行い、クラスタごとにプロットする

        Args:
            n_clusters (int, optional): クラスタ数. Default is 3.
        """
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(self.X)
        kmeans = KMeans(n_clusters=n_clusters, random_state=self.random_state, n_init="auto")
        
        kmeans.fit(X_scaled)
        self.kmeans_labels = kmeans.labels_
        
        print("KMeans法で予測したラベル:\n{}".format(self.kmeans_labels))

        # クラスタごとに色とマーカーを設定
        colors = ['red', 'blue', 'green']
        markers = ['^', 'o', 'v']  # ▲, ○, ▼

        for cluster in range(n_clusters):
            plt.scatter(X_scaled[self.kmeans_labels == cluster, 0], X_scaled[self.kmeans_labels == cluster, 1], 
                        c=colors[cluster], marker=markers[cluster])

        # クラスタの中心を黒い星でプロット
        plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 
                    marker='*', s=500, color='black')

        # 軸ラベルを追加
        plt.xlabel("First principal component", fontsize=12)
        plt.ylabel("Second principal component", fontsize=12)
        
        plt.legend()
        plt.show()
        
        print("実際のラベル:\n{}".format(self.y))
        
        for cluster in range(n_clusters):
            plt.scatter(X_scaled[self.y == cluster, 0], X_scaled[self.y == cluster, 1], 
                        c=colors[cluster], marker=markers[cluster], label=f"Cluster {cluster}")
        plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 
                    marker='*', s=500, color='black')
        plt.xlabel("First principal component", fontsize=12)
        plt.ylabel("Second principal component", fontsize=12)
        
    def plot_dendrogram(self, truncate=False, p=10):
        """Ward法を用いた階層的クラスタリングのデンドログラムを描画

        Args:
            truncate (bool): True の場合、デンドログラムを省略表示
            p (int): truncate=True の場合に表示するクラスタの数 (デフォルト: 5)
        """
        
        linkage_array = ward(self.X)  # Ward法によるクラスタリング

        # クラスタラベルを取得（3個に分割）
        self.ward_labels = fcluster(linkage_array, t=3, criterion='maxclust')

        # truncate=True の場合、一部のクラスタのみ表示
        if truncate:
            dendrogram(linkage_array, truncate_mode='lastp', p=p)
        else:
            dendrogram(linkage_array)

    def plot_dbscan(self, scaling=False, eps=0.6, min_samples=3):
        """DBSCAN を用いたクラスタリング結果をプロット

        Args:
            scaling (bool): True の場合、特徴量を標準化
            eps (float): クラスタリングの最大距離閾値
            min_samples (int): クラスタとみなす最小データ数
        """
        # スケーリング処理
        if scaling:
            X_scaled = StandardScaler().fit_transform(self.X)
        else:
            X_scaled = self.X

        # DBSCAN クラスタリング
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        clusters = dbscan.fit_predict(X_scaled) # 0, 1, -1
        self.dbscan_labels = clusters

        # Iris は品種が3種類 → クラスタ0,1,2 に割り当て、ノイズ(-1)も加える
        cluster_colors = {
            0: 'red',
            1: 'green',
            2: 'orange',
            -1: 'blue'
        }

        for cluster, color in cluster_colors.items():
            plt.scatter(
                X_scaled[clusters == cluster, 2],  # Petal Length
                X_scaled[clusters == cluster, 3],  # Petal Width
                c=color, edgecolors='k', s=60,
                label=f"Cluster {cluster}" if cluster != -1 else "Noise"
            )

        plt.xlabel("Feature 2 (Petal Length)")
        plt.ylabel("Feature 3 (Petal Width)")
        plt.title(f"DBSCAN Clustering (eps={eps}, min_samples={min_samples})")
        plt.legend()
        plt.show()

        print("Cluster Memberships:\n", clusters)

    def calc_all_ARI(self) -> None:
        """各クラスタリング手法のAdjusted Rand Index (ARI)を計算し表示する"""
        ari_results = {}

        if hasattr(self, "kmeans_labels"):
            ari_results["KMeans"] = adjusted_rand_score(self.y, self.kmeans_labels)

        if hasattr(self, "ward_labels"):
            ari_results["ward"] = adjusted_rand_score(self.y, self.ward_labels)

        if hasattr(self, "dbscan_labels"):
            ari_results["DBSCAN"] = adjusted_rand_score(self.y, self.dbscan_labels)

        print("\n=== Adjusted Rand Index (ARI) ===")
        for method, ari in ari_results.items():
            print(f"{method}: {ari:.3f}")

# データ分析の聞かれるポイント

# 元データ見る。pairplot見てわかることは？
# 練習問題2。一番いい結果は？なんで？どうしてこの手法が結果良かった？
# best_supervisedは5つの結果の平均を返す。学習済みモデルを渡すには？
# tsne, pca, nmfの結果を比較する.どれが一番いい？
# スケールがどれがいい？
# 思ったこと言ってよ
# 目的は一番精度が高いIrisの分類モデルを見つけること
# どの手法がいい？
# コードかデータ分析の穴がある方突っ込まれる
