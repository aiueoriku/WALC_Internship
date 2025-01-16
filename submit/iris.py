"""Irisデータセットを分析するモジュール"""

import pandas as pd
import seaborn as sns
from sklearn.datasets import load_iris


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