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


class AnalyzeIris:
    """Irisデータセットを分析するクラス"""

    # random_state=0 は再現性を担保するために設定
    # FIXME : random_stateを変えたい場合はどうしますか？この乱数では結果が悪かったので他の乱数を使いたい場合も出てくるかもしれません。
    # FIXED : random_stateをコンストラクタの引数として設定しました

    def __init__(self, random_state=0):  # self: インスタンス自身を指す
        """AnalyzeIrisクラスのコンストラクタ

        Attributes:
            data (pd.DataFrame): Irisデータセットのデータフレーム
            scores (dict): モデルのスコアを格納する辞書
            random_state (int): 乱数のシード

        """
        # FIXME: dataの中身がdfなので、それを明示した名前がいいと思います。df_dataとか
        # FIXED: data_dfに変更しました
        
        # self.data_df = self._load_iris_data()  # データを初期化時にロード
        self.data_df = self._load_iris_data()  # データを初期化時にロード
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
        

    def _load_iris_data(self):
        """Irisデータセットをロードして特徴量データフレームに変換
        
        Returns:
            pd.DataFrame: Irisデータセットの特徴量データフ
        """
        iris = load_iris()
        df_data = pd.DataFrame(iris.data, columns=iris.feature_names)
        return df_data

    def get(self):
        """Irisデータセットをロードしてデータフレームを返す（ラベルを追加）
        FIXME: 関数名とやっていることがあっていません。get関数でラベルを追加する挙動は使用する側が混乱します。
        Returns:
            pd.DataFrame: ラベルを含む新しいデータフレーム
        """
        if "label" not in self.data_df.columns:
            # 新しいデータフレームにラベルを追加
            self.data_df_with_label = self.data_df.copy()
            self.data_df_with_label["label"] = load_iris().target
            return self.data_df_with_label
        return self.data_df

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
        return self.data_df.corr()

    def pair_plot(self, diag_kind: str = "hist") -> sns.PairGrid:
        """ペアプロットを作成して表示

        Args:
            diag_kind (str): 対角成分のグラフの種類. Default is "hist".

        Returns:
            sns.PairGrid: 作成されたペアプロット
        """
        # if self.data_df is None:
        #     self.get()
        self.data_df_with_label["label"][self.data_df_with_label["label"] == 0] = "setosa"
        self.data_df_with_label["label"][self.data_df_with_label["label"] == 1] = "versicolor"
        self.data_df_with_label["label"][self.data_df_with_label["label"] == 2] = "virginica"
        return sns.pairplot(self.data_df_with_label, hue="label", diag_kind=diag_kind)

    def all_supervised(self, n_neighbors: int=4, n_splits: int = 5, shuffle: bool = True, random_state: int = 0) -> None:
        """複数の教師あり学習モデルを評価する

        Args:
            n_neighbors (int): Number of neighbors to use for KNeighborsClassifier. Default is 4.
            n_splits (int): Number of splits for KFold. Default is 5.
            shuffle (bool): Whether to shuffle the data. Default is True.
            random_state (int): Random seed. Default is 0.

        Returns:
            None
        """
        
        iris = load_iris()
        X = iris.data
        y = iris.target

        kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
        
        # スコアとモデルをリセット
        self.scores = {}
        self.trained_models = {}
        
        for model_name, model in self.models:
            print(f"=== {model_name} ===")

            self.scores[model_name] = []

            for train_index, test_index in kf.split(X):
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]

                model.fit(X_train, y_train)

                train_score = model.score(X_train, y_train)
                test_score = model.score(X_test, y_test)

                self.scores[model_name].append(test_score)
                self.trained_models[model_name] = model

                print(f"test score: {test_score:.3f}, train score: {train_score:.3f}")

    def get_supervised(self,n_splits: int = 5, shuffle: bool = True, random_state: int = 0):
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
        """
        
        if not self.scores:
            self.all_supervised()

        """
        else:
            all_supervisedの引数とget_supervisedの引数が異なる場合、get_supervisedの引数を使ってall_supervisedを呼び出すように変更する
        """

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
                feature_names = self.data_df.columns[:n_features]
                
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
        iris = load_iris()
        X = iris.data
        y = iris.target
        
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

        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

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
            n_components (int, optional): _description_. Defaults to 2.
        """
        iris = load_iris()
        X = iris.data
        y = iris.target

        # スケーリング前のPCA
        pca = PCA(n_components=n_components)
        pca.fit(X)
        X_pca = pca.transform(X)
        
        # plt.figure(figsize=(8, 8))
        # mglearn.discrete_scatter(X_pca[:, 0], X_pca[:, 1], y)
        # plt.legend(iris.target_names, loc="best")
        # plt.gca().set_aspect("equal")
        # plt.xlabel("First component")
        # plt.ylabel("Second component")
        # plt.ylim([X_pca[:, 1].min() - 0.5, X_pca[:, 1].max() + 0.5])
        # plt.xlim([X_pca[:, 0].min() - 0.5, X_pca[:, 0].max() + 0.5])
        
        # plt.matshow(pca.components_, cmap='viridis')
        # plt.yticks([0, 1], ["First component", "Second component"])
        # plt.colorbar()
        # plt.xticks(range(len(iris.feature_names)), iris.feature_names, rotation=60, ha='left')
        # plt.xlabel("Feature")
        # plt.ylabel("Principal components")
        

        # スケーリング後のPCA
        # scaler = MinMaxScaler()
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        pca_scaled = PCA(n_components=n_components)
        pca_scaled.fit(X_scaled)
        X_scaled_pca = pca_scaled.transform(X_scaled)

        
        plt.figure(figsize=(13, 8))
        mglearn.discrete_scatter(X_scaled_pca[:, 0], X_scaled_pca[:, 1], y)
        plt.legend(iris.target_names, loc="best")
        plt.gca().set_aspect("equal")
        plt.xlabel("First component")
        plt.ylabel("Second component")
        plt.ylim([X_scaled_pca[:, 1].min() - 0.5, X_scaled_pca[:, 1].max() + 0.5])
        plt.xlim([X_scaled_pca[:, 0].min() - 0.5, X_scaled_pca[:, 0].max() + 0.5])

        plt.matshow(pca_scaled.components_, cmap='viridis')
        plt.yticks([0, 1], ["First component", "Second component"])
        plt.colorbar()
        plt.xticks(range(len(iris.feature_names)), iris.feature_names, rotation=60, ha='left')
        plt.xlabel("Feature")
        plt.ylabel("PCA components")
        
        X_scaled_df = pd.DataFrame(X_scaled, columns=iris.feature_names)
        df_pca = pd.DataFrame(pca_scaled.components_, columns=iris.feature_names)
        
        return X_scaled_df, df_pca, pca_scaled
    

    def plot_nmf(self, n_components=2):
            """NMFによる次元削減を行い、2次元に削減したデータをプロットする

            Args:
                n_components (int, optional): _description_. Defaults to 2.
            """
            iris = load_iris()
            X = iris.data
            y = iris.target

            # スケーリング前のNMF
            nmf = NMF(n_components=n_components, init='random', random_state=0)
            nmf.fit(X)
            X_nmf = nmf.transform(X)
            
            plt.figure(figsize=(8, 8))
            mglearn.discrete_scatter(X_nmf[:, 0], X_nmf[:, 1], y)
            plt.legend(iris.target_names, loc="best")
            plt.gca().set_aspect("equal")
            plt.xlabel("First component")
            plt.ylabel("Second component")  
            plt.title("NMF")
            plt.ylim([X_nmf[:, 1].min() - 0.5, X_nmf[:, 1].max() + 0.5])
            plt.xlim([X_nmf[:, 0].min() - 0.5, X_nmf[:, 0].max() + 0.5])
            
            plt.matshow(nmf.components_, cmap='viridis')
            plt.yticks(range(n_components), [f"Component {i+1}" for i in range(n_components)])
            plt.colorbar()
            plt.xticks(range(len(iris.feature_names)), iris.feature_names, rotation=60, ha='left')
            plt.xlabel("Feature")
            plt.ylabel("NMF components")

            # スケーリング後のNMF
            scaler = StandardScaler()
            # scaler = MinMaxScaler()
            X_scaled = scaler.fit_transform(X)
            
            nmf_scaled = NMF(n_components=n_components, init='random', random_state=0)
            # nmf_scaled.fit(X_scaled)
            X_scaled_nmf = nmf_scaled.fit_transform(X_scaled - X_scaled.min())  # NMFは非負値のみ扱うため、最小値を調整



            plt.figure(figsize=(13, 8))
            mglearn.discrete_scatter(X_scaled_nmf[:, 0], X_scaled_nmf[:, 1], y)
            plt.legend(iris.target_names, loc="best")
            plt.gca().set_aspect("equal")
            plt.xlabel("First component")
            plt.ylabel("Second component")
            plt.title("NMF with StandardScaler")
            plt.ylim([X_scaled_nmf[:, 1].min()-0.5, X_scaled_nmf[:, 1].max() + 0.5])
            plt.xlim([X_scaled_nmf[:, 0].min()-0.5, X_scaled_nmf[:, 0].max() + 0.5])

            plt.matshow(nmf_scaled.components_, cmap='viridis')
            plt.yticks(range(n_components), [f"Component {i+1}" for i in range(n_components)])
            plt.colorbar()
            plt.xticks(range(len(iris.feature_names)), iris.feature_names, rotation=60, ha='left')
            plt.xlabel("Feature")
            plt.ylabel("NMF components")
            
            
            X_scaled_df = pd.DataFrame(X_scaled, columns=iris.feature_names)
            df_nmf = pd.DataFrame(nmf_scaled.components_, columns=iris.feature_names)
            
            return X_scaled_df, df_nmf, nmf_scaled
        
        