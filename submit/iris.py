"""Irisデータセットを分析するモジュール"""

import graphviz
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_iris
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
        self.data = self._load_iris_data()  # データを初期化時にロード
        self.data_with_label = None  # ラベル付きデータは初期化時にはNone
        self.scores = {}  # 各メソッドで共通の結果を格納
        self.trained_models = {}  # 学習済みモデルを格納
        self.models = [
            ("LogisticRegression", LogisticRegression(random_state=random_state)),
            ("LinearSVC", LinearSVC(random_state=random_state)),
            ("SVC", SVC(random_state=random_state)),
            ("DecisionTreeClassifier", DecisionTreeClassifier(random_state=random_state, max_depth=4)),
            ("KNeighborsClassifier", KNeighborsClassifier(n_neighbors=4)),
            ("LinearRegression", LinearRegression()),
            ("RandomForestClassifier", RandomForestClassifier(random_state=random_state)),
            ("GradientBoostingClassifier", GradientBoostingClassifier(random_state=random_state)),
            ("MLPClassifier", MLPClassifier(random_state=random_state))
        ]

    def _load_iris_data(self):
        """Irisデータセットをロードして特徴量データフレームに変換"""
        iris = load_iris()
        data = pd.DataFrame(iris.data, columns=iris.feature_names)
        return data

    def get(self):
        """Irisデータセットをロードしてデータフレームを返す（ラベルを追加）

        Returns:
            pd.DataFrame: ラベルを含む新しいデータフレーム
        """
        if "label" not in self.data.columns:
            # 新しいデータフレームにラベルを追加
            self.data_with_label = self.data.copy()
            self.data_with_label["label"] = load_iris().target
            return self.data_with_label
        return self.data

    def get_correlation(self):
        """データフレームの相関行列を計算

        Returns:
            pd.DataFrame: データフレームの相関行列

        """
        # if self.data is None:
        #     # FIXME: 全ての関数でこれをやっているなら修正が必要ですね
        #     self.get()
        # data_without_label = self.data.drop("label", axis=1)
        # return data_without_label.corr()
        return self.data.corr()

    def pair_plot(self, diag_kind: str = "hist") -> sns.PairGrid:
        """ペアプロットを作成して表示

        Args:
            diag_kind (str): 対角成分のグラフの種類. Default is "hist".

        Returns:
            sns.PairGrid: 作成されたペアプロット
        """
        # if self.data is None:
        #     self.get()
        self.data_with_label["label"][self.data_with_label["label"] == 0] = "setosa"
        self.data_with_label["label"][self.data_with_label["label"] == 1] = "versicolor"
        self.data_with_label["label"][self.data_with_label["label"] == 2] = "virginica"
        return sns.pairplot(self.data_with_label, hue="label", diag_kind=diag_kind)

    def all_supervised(self, n_neighbors: int=4, model_params: dict = None, n_splits: int = 5, shuffle: bool = True, random_state: int = 0) -> None:
        """複数の教師あり学習モデルを評価する

        Args:
            n_neighbors (int): Number of neighbors to use for KNeighborsClassifier. Default is 4.

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

    def get_supervised(self, model_params: dict = None, n_splits: int = 5, shuffle: bool = True, random_state: int = 0):
        """教師あり学習モデルの評価を行いpandas.DataFrameで返す

        Returns:
            pd.DataFrame: 教師あり学習モデルの評価結果

        TODO:
        これは方向性の問題なので修正が必ずしも必要ではありません。
        パラメータの変更などを行うことはないのですか？
        分析クラスなので、パラメータの変更を行なって再度分析したくなると思うのですが
        このget_supervisedの実装だと、パラメータを変更しても最初に実行した結果が上書きされないと思います。
        
        FIXED:
        パラメータを引数として受け取るように変更しました。
        """
        if not self.scores or model_params:
            self.all_supervised()

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
        TODO: class.__name__()で一応クラス名とってこれます。
        FIXED: class.__name__()でクラス名を取得するように変更しました。
        """
        for _, model in self.trained_models.items():
            if hasattr(model, "feature_importances_"):
                # モデルが提供する feature_importances_ に基づいて特徴量数を取得
                n_features = len(model.feature_importances_)
                
                # 特徴量名を調整
                feature_names = self.data.columns[:n_features]
                
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
                    feature_names=self.data_with_label.columns[:-1],
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
        
        iris = load_iris()
        X = iris.data
        y = iris.target
        
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        
        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            
            # スケーラーを適用しない場合の結果を出力
            model = LinearSVC(random_state=random_state)
            model.fit(X_train, y_train)
            
            train_score = model.score(X_train, y_train)
            test_score = model.score(X_test, y_test)
            print(f"Original: train_score: {train_score:.3f}   test_score: {test_score:.3f}")
            
            fig, ax = plt.subplots(1,5, figsize=(13, 4))
            ax[0].scatter(X_train[:,0], X_train[:,1], c='blue', marker='o', label='Train')
            ax[0].scatter(X_test[:,0], X_test[:,1], c='red', marker='^', label='Test')
            ax[0].set_title("Original")
            ax[0].set_xlabel("sepal length (cm)")
            ax[0].set_ylabel("sepal width (cm)")
            ax[0].legend()
            
            
            
            
            # for scaler in self.scalers:
            for i, scaler in enumerate(self.scalers):
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
                model = LinearSVC(random_state=random_state)
                model.fit(X_train_scaled, y_train)
                
                train_score = model.score(X_train_scaled, y_train)
                test_score = model.score(X_test_scaled, y_test)
                
                print(f"{scaler.__class__.__name__}: train_score: {train_score:.3f}   test_score: {test_score:.3f}")
                
                ax[i+1].scatter(X_train_scaled[:,0], X_train_scaled[:,1], c='blue', marker='o', label='Train')
                ax[i+1].scatter(X_test_scaled[:,0], X_test_scaled[:,1], c='red', marker='^', label='Test')
                ax[i+1].set_title(scaler.__class__.__name__)
                ax[i+1].legend()

                
            print("="*50)
            
            # 見た目がちょっと違うのと、printの順番が違うのは研究室で聞く