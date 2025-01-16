"""Irisデータセットを分析するモジュール"""

import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression. LinearRegression
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression

class AnalyzeIris:
    """Irisデータセットを分析するクラス """

    def __init__(self):
        """AnalyzeIrisクラスのコンストラクタ

        Attributes:
            data (pd.DataFrame): Irisデータセットのデータフレーム

        """
        self.data = None

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
            diag_kind (str): 対角線に表示するプロットの種類 デフォルトは'hist'

        Returns:
            sns.PairGrid: 作成されたペアプロット

        Raises:
            ValueError: データがロードされていない場合に発生
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
        
        models = [
            ("LogisticRegression", LogisticRegression()),
            ("LinearSVC", LinearSVC()),
            ("SVC", SVC()),
            ("DecisionTreeClassifier", DecisionTreeClassifier()),
            ("KNeighborsClassifier", KNeighborsClassifier(n_neighbors=n_neighbors)),
            ("LinearRegression", LinearRegression()),
            ("RandomForestClassifier", RandomForestClassifier()),
            ("GradientBoostingClassifier", GradientBoostingClassifier()),
            ("MLPClassifier", MLPClassifier())
        ]
        
        for name, model in models:
            print(f"=== {name} ===")
            
            for train_index, test_index in kf.split(X):
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]
                
                model.fit(X_train, y_train)
                
                train_score = model.score(X_train, y_train)
                test_score = model.score(X_test, y_test)
                
                print(f"test score: {test_score:.3f}, train score: {train_score:.3f}")