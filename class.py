class Myclass:
    pass

i = Myclass()

# attribute(変数みたいなもの)を追加
i.value = 5
print(i.value)

# attributeはインスタンスごとに作る必要がある
j = Myclass()
try:
    print(j.value)
except AttributeError as e:
    print(e)

# __init__メソッドはattributeを自動で追加する
class Prism:
    def __init__(self, width, height, depth):
        self.width = width
        self.height = height
        self.depth = depth

    def content(self):
        return self.width * self.height * self.depth

p1 = Prism(10, 20, 30)
print(p1.content()) # コンストラクタの引数が自動で入る
print(p1.width) # attribute

p1.width = "a"
print(p1.content()) # 意図しない結果を招く場合がある

