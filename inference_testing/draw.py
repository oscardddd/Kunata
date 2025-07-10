import matplotlib.pyplot as plt
import numpy as np


def draw_wikiqa():
    # 示例数据
    labels = ['qwen3-0.6b', 'qwen3-4b', 'qwen3-8b']  # x轴类别
    group1 = [8, 14, 36]     # 第一组
    group2 = [20.52, 29.33, 45.3]     # 第二组

    x = np.arange(len(labels))  # x轴的位置 [0, 1, 2]
    width = 0.25                # 每个柱子的宽度

    # 创建柱状图
    plt.bar(x - width, group1, width, label='EM', color='skyblue')
    plt.bar(x, group2, width, label='F1', color='orange')
    # plt.bar(x + width, group3, width, label='组3', color='green')

    # 添加标签
    plt.xlabel('Model')
    plt.ylabel('Score (%)')
    plt.title('Model performance on 2wikimqa with respect to size')
    plt.xticks(x, labels)
    plt.legend()

    # 显示图形
    plt.savefig("./size_2wikimqa_performance.jpg")
    

# draw_wikiqa()

import matplotlib.pyplot as plt
import numpy as np


def draw_wikiqa():
    # 示例数据
    labels = ['qwen3-0.6b', 'qwen3-4b', 'qwen3-8b']  # x轴类别
    group1 = [8, 14, 36]     # 第一组
    group2 = [20.52, 29.33, 45.3]     # 第二组

    x = np.arange(len(labels))  # x轴的位置 [0, 1, 2]
    width = 0.25                # 每个柱子的宽度

    # 创建柱状图
    plt.bar(x - width, group1, width, label='EM', color='skyblue')
    plt.bar(x, group2, width, label='F1', color='orange')
    # plt.bar(x + width, group3, width, label='组3', color='green')

    # 添加标签
    plt.xlabel('Model')
    plt.ylabel('Score (%)')
    plt.title('Model performance with respect to size')
    plt.xticks(x, labels)
    plt.legend()

    # 显示图形
    plt.savefig("./size_performance.jpg")
    
# 0.6b:
# hf (pretrained=Qwen/Qwen3-0.6B), gen_kwargs: (None), limit: 50.0, num_fewshot: None, batch_size: 8
# |Tasks|Version|     Filter     |n-shot|  Metric   |   |Value|   |Stderr|
# |-----|------:|----------------|-----:|-----------|---|----:|---|-----:|
# |gsm8k|      3|flexible-extract|     5|exact_match|↑  |  0.4|±  |  0.07|
# |     |       |strict-match    |     5|exact_match|↑  |  0.4|±  |  0.07|


# 4b: 
# # hf (pretrained=Qwen/Qwen3-4B), gen_kwargs: (None), limit: 50.0, num_fewshot: None, batch_size: 8
# |Tasks|Version|     Filter     |n-shot|  Metric   |   |Value|   |Stderr|
# |-----|------:|----------------|-----:|-----------|---|----:|---|-----:|
# |gsm8k|      3|flexible-extract|     5|exact_match|↑  | 0.86|±  |0.0496|
# |     |       |strict-match    |     5|exact_match|↑  | 0.86|±  |0.0496|

# 8b: 
# hf (pretrained=Qwen/Qwen3-8B), gen_kwargs: (None), limit: 50.0, num_fewshot: None, batch_size: 8
# |Tasks|Version|     Filter     |n-shot|  Metric   |   |Value|   |Stderr|
# |-----|------:|----------------|-----:|-----------|---|----:|---|-----:|
# |gsm8k|      3|flexible-extract|     5|exact_match|↑  | 0.92|±  |0.0388|
# |     |       |strict-match    |     5|exact_match|↑  | 0.92|±  |0.0388|



import numpy as np
import matplotlib.pyplot as plt

def draw_math():
    # 模型名称（x轴标签）
    labels = ['Qwen3-0.6B', 'Qwen3-4B', 'Qwen3-8B']
    # EM 分数（单组数据）
    scores = [0.4, 0.86, 0.92]  # 小数表示百分比，可乘100

    x = np.arange(len(labels))  # x轴位置索引
    width = 0.6                 # 每个柱子的宽度

    # 创建柱状图
    plt.figure(figsize=(8, 6))
    plt.bar(x, [s * 100 for s in scores], width, color='skyblue')

    # 添加标签
    plt.xlabel('Model')
    plt.ylabel('EM Score (%)')
    plt.title('Math Task: EM Scores by Model Size')
    plt.xticks(x, labels)
    plt.ylim(0, 100)  # 设置 y 轴范围为 0-100%

    # # 每个柱子顶部标数值
    # for i, score in enumerate(scores):
    #     plt.text(x[i], score * 100 + 1, f'{score:.0%}', ha='center')

    # 显示图形
    plt.tight_layout()
    plt.savefig('./math_size_em_score.jpg')
    plt.show()



draw_math()
