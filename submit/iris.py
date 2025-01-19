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
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier, export_graphviz


class AnalyzeIris:
    """Irisデータセットを分析するクラス """

    # random_state=0 は再現性を担保するために設定
    models = [
            ("LogisticRegression", LogisticRegression(random_state=0)),
            ("LinearSVC", LinearSVC(random_state=0)),
            ("SVC", SVC(random_state=0)),
            ("DecisionTreeClassifier", DecisionTreeClassifier(random_state=0, max_depth=4)),
            ("KNeighborsClassifier", KNeighborsClassifier(n_neighbors=4)),
            ("LinearRegression", LinearRegression()),
            ("RandomForestClassifier", RandomForestClassifier(random_state=0)),
            ("GradientBoostingClassifier", GradientBoostingClassifier(random_state=0)),
            ("MLPClassifier", MLPClassifier(random_state=0))
        ] # class variable


    def __init__(self): # self: インスタンス自身を指す
        """AnalyzeIrisクラスのコンストラクタ

        Attributes:
            data (pd.DataFrame): Irisデータセットのデータフレーム
            scores (dict): モデルのスコアを格納する辞書

        """
        self.data = pd.DataFrame()
        self.scores = {} # 各メソッドで共通の結果を格納
        self.trained_models = {} # 学習済みモデルを格納

    def get(self):
        """Irisデータセットをロードしてデータフレームに変換

        Returns:
            pd.DataFrame: Irisデータセットを含むデータフレーム
        """
        iris = load_iris()
        self.data = pd.DataFrame(iris.data, columns=iris.feature_names) # 明日研究室で聞いてみよう
        self.data["label"] = iris.target
        return self.data

    def get_correlation(self):
        """データフレームの相関行列を計算

        Returns:
            pd.DataFrame: データフレームの相関行列

        """
        if self.data is None:
            self.get()
        data_without_label = self.data.drop("label", axis=1)
        return data_without_label.corr()

    def pair_plot(self, diag_kind: str = "hist") -> sns.PairGrid:
        """ペアプロットを作成して表示

        Args:
            diag_kind (str): 対角成分のグラフの種類. Default is "hist".

        Returns:
            sns.PairGrid: 作成されたペアプロット
        """
        if self.data is None:
            self.get()
        self.data["label"][self.data["label"]==0] = "setosa"
        self.data["label"][self.data["label"]==1] = "versicolor"
        self.data["label"][self.data["label"]==2] = "virginica"
        return sns.pairplot(self.data, hue="label", diag_kind=diag_kind)    

    def all_supervised(self, n_neighbors: int = 4) -> None:
        """複数の教師あり学習モデルを評価する

        Args:
            n_neighbors (int): Number of neighbors to use for KNeighborsClassifier. Default is 4.

        Returns:
            None
        """
        iris = load_iris()
        X = iris.data
        y = iris.target

        kf = KFold(n_splits=5, shuffle=True, random_state=0)

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

    def get_supervised(self):
        """教師あり学習モデルの評価を行いpandas.DataFrameで返す

        Returns:
            pd.DataFrame: 教師あり学習モデルの評価結果
        """
        if not self.scores:
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
        mean = df_results.mean() # 各列の平均値を計算
        best_method = mean.idxmax() # 最大値を持つ列の名前を取得
        best_score = mean.max() # 最大値を取得
        return best_method, best_score
    
    def plot_feature_importances_all(self):
        """全てのモデルの特徴量の重要度をプロットする"""
        for model_name, model in self.trained_models.items():
            if hasattr(model, 'feature_importances_'):
                n_features = len(self.data.columns) - 1
                plt.barh(range(n_features), model.feature_importances_, align="center")
                plt.yticks(np.arange(n_features), self.data.columns[:-1])
                plt.xlabel(f"Feature importance: {model_name}")
                plt.show()
                print("\n")

    def visualize_decision_tree(self):
        # print("=== DecisionTreeClassifier ===")
        """決定木の可視化"""
        for model_name, model in self.trained_models.items():
            if model_name == "DecisionTreeClassifier":
                export_graphviz(model, out_file=f"{model_name}.dot", feature_names=self.data.columns[:-1], class_names=["setosa", "versicolor", "virginica"], filled=True, rounded=True)
                with open(f"{model_name}.dot") as f:
                    dot_graph = f.read()
                graph = graphviz.Source(dot_graph)
                return graph