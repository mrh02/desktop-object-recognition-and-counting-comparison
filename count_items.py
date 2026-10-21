# 暂时用这个列表模拟模型识别出的物品
items = ["杯子", "鼠标", "杯子", "键盘", "鼠标","杯子"]

# 创建一个空字典，保存每种物品的数量
counts = {}

# 依次取出列表中的物品
for item in items:
    if item in counts:
        counts[item] += 1
    else:
        counts[item] = 1

# 显示统计结果
for name, number in counts.items():
    print(f"{name}：{number}个")