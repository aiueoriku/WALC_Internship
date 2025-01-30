"""Irisデータセットを分析するモジュール"""

# import graphviz
import matplotlib.pyplot as plt
import mglearn
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.decomposition import NMF, PCA
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler, Normalizer, RobustScaler, StandardScaler
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, DBSCAN
from scipy.cluster.hierarchy import dendrogram, ward



class AnalyzeIris:
    """Irisデータセットを分析するクラス"""

    # random_state=0 は再現性を担保するために設定
    # FIXME : random_stateを変えたい場合はどうしますか？この乱数では結果が悪かったので他の乱数を使いたい場合も出てくるかもしれません。
    # FIXED : random_stateをコンストラクタの引数として設定しました

    def __init__(self, random_state=0):  # self: インスタンス自身を指す
        """コンストラクタ

        Args:
            random_state (int, optional): 乱数のシード. Defaults to 0.
            
        """
        # FIXME: dataの中身がdfなので、それを明示した名前がいいと思います。df_dataとか
        # FIXED: data_dfに変更しました
        
        # self.data_df = self._load_iris_data()  # データを初期化時にロード
        # self.data_df = self._load_iris_data()  # データを初期化時にロード
        
        # データセットの読み込み
        iris = load_iris()
        self.X = iris.data # 特徴量
        self.y = iris.target # ラベル
        self.feature_names = iris.feature_names # 特徴量名
        self.target_names = iris.target_names # ラベル名
        
        self.data_df_with_label = None  # ラベル付きデータは初期化時にはNone
        self.scores = {}  # 各メソッドで共通の結果を格納
        self.random_state = random_state # random_stateをインスタンス変数として保持
        self.trained_models = {}  # 学習済みモデルを格納
        self.models = [
            ("LogisticRegression", LogisticRegression(random_state=random_state)), # FIXME: random_stateはselfで持っていた方がいいです。他でも使うかもしれません。
            ("LinearSVC", LinearSVC(random_state=random_state)), # FIXED: random_stateをselfで持つように変更しました
            ("SVC", SVC(random_state=random_state)),
            ("DecisionTreeClassifier", DecisionTreeClassifier(random_state=random_state, max_depth=4)),
            ("KNeighborsClassifier", KNeighborsClassifier(n_neighbors=4)),
            ("LinearRegression", LinearRegression()),
            ("RandomForestClassifier", RandomForestClassifier(random_state=random_state)),
            ("GradientBoostingClassifier", GradientBoostingClassifier(random_state=random_state)),
            ("MLPClassifier", MLPClassifier(random_state=random_state))
        ]
        

    # def _load_iris_data(self):
    #     """Irisデータセットをロードして特徴量データフレームに変換
        
    #     Returns:
    #         pd.DataFrame: Irisデータセットの特徴量データフ
    #     """
    #     iris = load_iris()
    #     df_data = pd.DataFrame(iris.data, columns=iris.feature_names)
    #     return df_data

    # def get(self):
    #     """Irisデータセットをロードしてデータフレームを返す（ラベルを追加）
    #     FIXME: 関数名とやっていることがあっていません。get関数でラベルを追加する挙動は使用する側が混乱します。
    #     Returns:
    #         pd.DataFrame: ラベルを含む新しいデータフレーム
    #     """
    #     if "label" not in self.data_df.columns:
    #         # 新しいデータフレームにラベルを追加
    #         self.data_df_with_label = self.data_df.copy()
    #         self.data_df_with_label["label"] = load_iris().target
    #         return self.data_df_with_label
    #     return self.data_df
    
    def get(self):
        """ラベル付のデータフレームを返す

        Returns:
            pd.DataFrame: ラベルを含むデータフレーム
        """
        data_df = pd.DataFrame(self.X, columns=self.feature_names)
        if "label" not in data_df.columns:
            # データフレームにラベルを追加
            self.data_df_with_label = data_df.copy()
            self.data_df_with_label["label"] = self.y
            return self.data_df_with_label

    def get_correlation(self):
        """データフレームの相関行列を計算

        Returns:
            pd.DataFrame: データフレームの相関行列

        """
        # if self.data_df is None:
        #     # FIXME: 全ての関数でこれをやっているなら修正が必要ですね
        #     self.get()
        # data_without_label = self.data_df.drop("label", axis=1)
        # return data_without_label.corr()
        data_df = pd.DataFrame(self.X, columns=self.feature_names)
        
        
        return data_df.corr()

    def pair_plot(self, diag_kind: str = "hist") -> sns.PairGrid:
        """ペアプロットを表示

        Args:
            diag_kind (str): 対角成分のグラフの種類. Default is "hist".

        Returns:
            sns.PairGrid: ペアプロット
        """
        # if self.data_df is None:
        #     self.get()
        self.data_df_with_label["label"][self.data_df_with_label["label"] == 0] = "setosa"
        self.data_df_with_label["label"][self.data_df_with_label["label"] == 1] = "versicolor"
        self.data_df_with_label["label"][self.data_df_with_label["label"] == 2] = "virginica"
        return sns.pairplot(self.data_df_with_label, hue="label", diag_kind=diag_kind)

    def all_supervised(self, n_neighbors = 4, n_splits: int = 5, shuffle: bool = True, random_state: int = 0) -> None:
        """複数の教師あり学習モデルを評価する

        Args:
            n_splits (int): KFoldの分割数
            shuffle (bool): データをシャッフルするかどうか
            random_state (int): 乱数シード

        Returns:
            None
        """
        
        # iris = load_iris()
        # X = iris.data
        # y = iris.target

        kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
        
        # スコアとモデルをリセット
        self.scores = {}
        self.trained_models = {}
        
        # 引数をget_supervisedからアクセスできるようにする
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state
        
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

    def get_supervised(self,n_splits: int = 5, shuffle: bool = True, random_state: int = 0) -> pd.DataFrame:
        """教師あり学習モデルの評価を行いpandas.DataFrameで返す
        FIXME: 入力のmodel_paramsは使われていません。この関数の中で使う予定はありますか？また、各引数の説明を書きましょう。
        FIXED: model_paramsを削除しました
        
        Args:
            n_splits (int): Number of splits for KFold.
            shuffle (bool): Whether to shuffle the data.
            random_state (int): Random seed. Default is 0.
        
        Returns:
            pd.DataFrame: 教師あり学習モデルの評価結果
            

        FIXED:
        パラメータを引数として受け取るように変更しました。
        FIXME: 引数にとったパラメータは使われていますか？この実装だとパラメータを変更しても、結果は変わらないと思います。
        FIXED: 引数を使ってKFoldの分割数、シャッフルの有無、乱数シードを変更できるようにしました。
        """
        
        if not self.scores:
            # 評価が行われていない場合、現在の引数で all_supervised を実行
            self.all_supervised(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
        else:
            # 既にスコアがある場合、引数が異なるかを確認し、異なれば再実行
            if self.n_splits != n_splits or self.shuffle != shuffle or self.random_state != random_state:
                print("引数が変更されたため、再評価を行います。")
                self.all_supervised(n_splits=n_splits, shuffle=shuffle, random_state=random_state)

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
                
                # 特徴量名を調整
                data_df = pd.DataFrame(self.X, columns=self.feature_names)
                feature_names = data_df.columns[:n_features]
                
                # プロット
                plt.barh(range(n_features), model.feature_importances_, align="center")
                plt.yticks(np.arange(n_features), feature_names)
                plt.xlabel(f"Feature importance: {model.__class__.__name__}")
                plt.title(f"Feature Importance of {model.__class__.__name__}")
                plt.show()

    def visualize_decision_tree(self):
        # print("=== DecisionTreeClassifier ===")
        """決定木の可視化"""
        for model_name, model in self.trained_models.items():
            if model_name == "DecisionTreeClassifier":
                export_graphviz(
                    model,
                    out_file=f"{model_name}.dot",
                    feature_names=self.data_df_with_label.columns[:-1],
                    class_names=["setosa", "versicolor", "virginica"],
                    filled=True,
                    rounded=True,
                )
                with open(f"{model_name}.dot") as f:
                    dot_graph = f.read()
                graph = graphviz.Source(dot_graph)
                return graph

    def plot_scaled_data(self, n_splits: int = 5, random_state: int = 0):
        """5-foldでそれぞれの要素に対するスケーリングとLinearSVCの結果を出力する
        """
        
        self.scalers = [MinMaxScaler(), 
                        StandardScaler(), 
                        RobustScaler(), 
                        Normalizer()]
        # self.scores_scaled = {}
        # FIXME: 毎回irisデータをロードしているので、一度ロードしておいてそれを使うようにしましょう。コンストラクタでロードしたものは使わないのですか？
        # FIXED: コンストラクタでロードしたものを使うように変更しました
        
        # iris = load_iris()
        # X = iris.data
        # y = iris.target
        
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        
        feature_names = ["sepal length (cm)", "sepal width (cm)", "petal length (cm)", "petal width (cm)"]
        plot_combinations = [
            (0, 1),  # sepal length vs sepal width
            (0, 2),  # sepal length vs petal length
            (0, 3),  # sepal length vs petal width
            (1, 2),  # sepal width vs petal length
            (1, 3),  # sepal width vs petal width
            (2, 3),  # petal length vs petal width
        ]

        for train_index, test_index in kf.split(self.X):
            X_train, X_test = self.X[train_index], self.X[test_index]
            y_train, y_test = self.y[train_index], self.y[test_index]

            # スケーラーを適用しない場合の結果を出力
            model = LinearSVC(random_state=random_state, max_iter=10000)
            model.fit(X_train, y_train)

            train_score = model.score(X_train, y_train)
            test_score = model.score(X_test, y_test)
            print(f"Original: test score: {test_score:.3f}   train score: {train_score:.3f}")

            # プロット用の設定
            fig, ax = plt.subplots(6, 5, figsize=(13, 15))  # 6行5列のプロット
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

                model = LinearSVC(random_state=random_state, max_iter=10000)
                model.fit(X_train_scaled, y_train)

                train_score = model.score(X_train_scaled, y_train)
                test_score = model.score(X_test_scaled, y_test)

                print(f"{scaler.__class__.__name__}: test score: {test_score:.3f}   train score: {train_score:.3f}")

                # 各スケーリング手法での散布図
                for i, (x_idx, y_idx) in enumerate(plot_combinations):
                    ax[i * 5 + (j + 1)].scatter(X_train_scaled[:, x_idx], X_train_scaled[:, y_idx], c='blue', marker='o', label='Train')
                    ax[i * 5 + (j + 1)].scatter(X_test_scaled[:, x_idx], X_test_scaled[:, y_idx], c='red', marker='^', label='Test')
                    ax[i * 5 + (j + 1)].set_title(f"{scaler.__class__.__name__}")
                    ax[i * 5 + (j + 1)].legend()

            plt.tight_layout()
            plt.show()  # ループごとにプロットを表示
            plt.close(fig)  # メモリの解放

            print("=" * 50)
            
    def plot_pca(self, n_components=2):
        """PCAによる次元削減を行い、2次元に削減したデータをプロットする

        Args:
            n_components (int, optional): 次元削減後の次元数. Defaults to 2.
            
        Returns:
            X_scaled_df (DataFrame): スケーリング後のデータ
            df_pca (DataFrame): PCAによる次元削減後のデータ
            pca_scaled (PCA): スケーリング後のPCAモデル
        """
        # iris = load_iris()
        # X = iris.data
        # y = iris.target

        # スケーリング前のPCA
        pca = PCA(n_components=n_components)
        pca.fit(self.X)
        X_pca = pca.transform(self.X)
        
        # plt.figure(figsize=(8, 8))
        # mglearn.discrete_scatter(X_pca[:, 0], X_pca[:, 1], self.y)
        # plt.legend(self.target_names, loc="best")
        # plt.gca().set_aspect("equal")
        # plt.xlabel("First component")
        # plt.ylabel("Second component")
        # plt.ylim([X_pca[:, 1].min() - 0.5, X_pca[:, 1].max() + 0.5])
        # plt.xlim([X_pca[:, 0].min() - 0.5, X_pca[:, 0].max() + 0.5])
        
        # plt.matshow(pca.components_, cmap='viridis')
        # plt.yticks([0, 1], ["First component", "Second component"])
        # plt.colorbar()
        # plt.xticks(range(len(self.feature_names)), self.feature_names, rotation=60, ha='left')
        # plt.xlabel("Feature")
        # plt.ylabel("Principal components")
        

        # スケーリング後のPCA
        # scaler = MinMaxScaler()
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(self.X)
        
        pca_scaled = PCA(n_components=n_components)
        pca_scaled.fit(X_scaled)
        X_scaled_pca = pca_scaled.transform(X_scaled)

        
        plt.figure(figsize=(13, 8))
        mglearn.discrete_scatter(X_scaled_pca[:, 0], X_scaled_pca[:, 1], self.y)
        plt.legend(self.target_names, loc="best")
        plt.gca().set_aspect("equal")
        plt.xlabel("First component")
        plt.ylabel("Second component")
        plt.ylim([X_scaled_pca[:, 1].min() - 0.5, X_scaled_pca[:, 1].max() + 0.5])
        plt.xlim([X_scaled_pca[:, 0].min() - 0.5, X_scaled_pca[:, 0].max() + 0.5])

        plt.matshow(pca_scaled.components_, cmap='viridis')
        plt.yticks([0, 1], ["First component", "Second component"])
        plt.colorbar()
        plt.xticks(range(len(self.feature_names)), self.feature_names, rotation=60, ha='left')
        plt.xlabel("Feature")
        plt.ylabel("PCA components")
        
        X_scaled_df = pd.DataFrame(X_scaled, columns=self.feature_names)
        df_pca = pd.DataFrame(pca_scaled.components_, columns=self.feature_names)
        
        return X_scaled_df, df_pca, pca_scaled
    

    def plot_nmf(self, n_components=2):
            """NMFによる次元削減を行い、2次元に削減したデータをプロットする

            Args:
                n_components (int, optional): 次元削減後の次元数. Defaults to 2.
            
            Returns:
                X_scaled_df (DataFrame): スケーリング後のデータ
                df_nmf (DataFrame): nmfによる次元削減後のデータ
                nmf_scaled (nmf): スケーリング後のnmfモデル
            """
            # iris = load_iris()
            # X = iris.data
            # y = iris.target

            # スケーリング前のNMF
            nmf = NMF(n_components=n_components, init='random', random_state=0)
            nmf.fit(self.X)
            X_nmf = nmf.transform(self.X)
            
            plt.figure(figsize=(8, 8))
            mglearn.discrete_scatter(X_nmf[:, 0], X_nmf[:, 1], self.y)
            plt.legend(self.target_names, loc="best")
            plt.gca().set_aspect("equal")
            plt.xlabel("First component")
            plt.ylabel("Second component")  
            plt.title("NMF")
            plt.ylim([X_nmf[:, 1].min() - 0.5, X_nmf[:, 1].max() + 0.5])
            plt.xlim([X_nmf[:, 0].min() - 0.5, X_nmf[:, 0].max() + 0.5])
            
            plt.matshow(nmf.components_, cmap='viridis')
            plt.yticks(range(n_components), [f"Component {i+1}" for i in range(n_components)])
            plt.colorbar()
            plt.xticks(range(len(self.feature_names)), self.feature_names, rotation=60, ha='left')
            plt.xlabel("Feature")
            plt.ylabel("NMF components")

            # スケーリング後のNMF
            scaler = StandardScaler()
            # scaler = MinMaxScaler()
            X_scaled = scaler.fit_transform(self.X)
            
            nmf_scaled = NMF(n_components=n_components, init='random', random_state=0)
            # nmf_scaled.fit(X_scaled)
            X_scaled_nmf = nmf_scaled.fit_transform(X_scaled - X_scaled.min())  # NMFは非負値のみ扱うため、最小値を調整



            plt.figure(figsize=(13, 8))
            mglearn.discrete_scatter(X_scaled_nmf[:, 0], X_scaled_nmf[:, 1], self.y)
            plt.legend(self.target_names, loc="best")
            plt.gca().set_aspect("equal")
            plt.xlabel("First component")
            plt.ylabel("Second component")
            plt.title("NMF with StandardScaler")
            plt.ylim([X_scaled_nmf[:, 1].min()-0.5, X_scaled_nmf[:, 1].max() + 0.5])
            plt.xlim([X_scaled_nmf[:, 0].min()-0.5, X_scaled_nmf[:, 0].max() + 0.5])

            plt.matshow(nmf_scaled.components_, cmap='viridis')
            plt.yticks(range(n_components), [f"Component {i+1}" for i in range(n_components)])
            plt.colorbar()
            plt.xticks(range(len(self.feature_names)), self.feature_names, rotation=60, ha='left')
            plt.xlabel("Feature")
            plt.ylabel("NMF components")
            
            
            X_scaled_df = pd.DataFrame(X_scaled, columns=self.feature_names)
            df_nmf = pd.DataFrame(nmf_scaled.components_, columns=self.feature_names)
            
            return X_scaled_df, df_nmf, nmf_scaled
        
    def plot_tsne(self, random_state=0):
        """t-SNEによる次元削減を行い、2次元に削減したデータをプロットする

        Args:
            random_state (int, optional): 乱数のシード. Defaults to 0.
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
    
    def plot_k_means(self, n_clusters=3, random_state=0):
        """KMeans法によるクラスタリングを行い、クラスタごとにプロットする

        Args:
            n_clusters (int, optional): クラスタ数. Defaults to 3.
            random_state (int, optional): 乱数のシード. Defaults to 0.
        """
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(self.X)
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto")
        
        kmeans.fit(X_scaled)
        
        print("KMeans法で予測したラベル:\n{}".format(kmeans.labels_))

        # クラスタごとに色とマーカーを設定
        colors = ['red', 'blue', 'green']
        markers = ['^', 'o', 'v']  # ▲, ○, ▼

        

        for cluster in range(n_clusters):
            plt.scatter(X_scaled[kmeans.labels_ == cluster, 0], X_scaled[kmeans.labels_ == cluster, 1], 
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

        # クラスタごとの色を手動設定
        cluster_colors = {0: 'red', 1: 'green', -1: 'blue'}

        for cluster, color in cluster_colors.items():
            plt.scatter(
                X_scaled[clusters == cluster, 2], X_scaled[clusters == cluster, 3], 
                c=color, edgecolors='k', s=60, label=f"Cluster {cluster}" if cluster != -1 else "Noise"
            )

        plt.xlabel("Feature 2 (Petal Length)")
        plt.ylabel("Feature 3 (Petal Width)")
        plt.title(f"DBSCAN Clustering (eps={eps}, min_samples={min_samples})")
        plt.legend()
        plt.show()

        print("Cluster Memberships:\n", clusters)

    def calc_ARI(self):
        """KMeans, Dendrogram, DBSCANのARIを計算"""
