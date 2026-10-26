# 图片清点对比实验

| 图片 | 人工答案 | YOLO结果 | 大模型结果 | YOLO计数正确 | 大模型计数正确 | API耗时 | Token用量 |
| 01_simple.jpg | mouse: 1 | mouse: 1 | mouse: 1 | 是 | 是 | 3.14秒 | 2599 |
| 02_simple.jpg | cup: 1   | toilet: 1| cup: 1   | 否 | 是 | 2.09秒 | 2599 |
| 03_simple.jpg | cup: 2, mouse: 1 | cup: 2, mouse: 1 | cup: 2, mouse: 1 | 是 | 是 | 2.92秒 | 2607 |
| 04_multiple.jpg| cup: 1, mouse: 1, scissors: 1 | cup: 1, mouse: 1, scissors: 1 | cup: 1, mouse: 1, scissors: 1 | 是 | 是 | 3.90秒 | 2616 |
| 05_multiple.jpg | cup: 1, mouse: 1, scissors: 1 | cup: 1, mouse: 1, scissors: 2 | cup: 1, mouse: 1, scissors: 1 | 否 | 是 | 3.52秒 | 2616 |
| 06_clutter.jpg | cup: 1, mouse: 1, scissors: 1 | mouse: 2, scissors: 1 | cup: 1, mouse: 1, scissors: 1 | 否 | 是 | 3.60秒 | 2642 |
| 07_occlusion.jpg | cup: 1, mouse: 1, scissors: 1 | mouse: 2, scissors: 3 | cup: 1, mouse: 1, scissors: 1 | 否 | 是 | 3.47秒 | 2616 |
## 02_simple.jpg：杯子类别误识别

YOLO将杯子误识别为toilet，置信度为0.93；
大模型识别为cup，与人工答案一致。

图片为俯拍视角，杯子侧面特征不明显，光线较暗。
这些因素可能与误识别有关，但尚未通过控制变量实验验证。
高置信度不代表分类一定正确。

03_simple.jpg是开发期间使用的验证图片，
此处记录重复验证结果，不作为独立泛化能力的证据。
曾用含充电宝的图片进行探索测试。
04_multiple.jpg
由于充电宝不属于当前YOLO预训练模型的支持类别，
该图片不纳入支持类别范围内的正式对比统计。
05_multiple.jpg
YOLO将包含线缆和鼠标的区域额外识别为剪刀，置信度0.30，
导致剪刀多计1件，错误检测框也传入了SAM。