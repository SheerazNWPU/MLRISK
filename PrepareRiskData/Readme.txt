

【风险分析整体流程】
① 训练神经网络
② 数据过一遍神经网络，获得特征向量
③ 特征向量转为风险特征（本文件夹中代码所做的）
④ 风险特征训练风险模型
⑤ 风险模型反馈神经网络


【整体架构】（用神经网络输出的向量生成 risk_dataset、泛化规则部分）
get_risk_dataset  # 主函数
    |
    |—— get_risk_info  # 用于生成 risk_dataset 给天伟风险分析
    |       |—— get_distance
    |       |—— get_knn_count
    |
    |—— get_one_risk_info  # 用于生成泛化特征，给单边决策树生成泛化规则
            |—— get_one_distance
            |—— get_one_knn_count


【代码设置】
所有设置都在 get_risk_dataset 里

输入文件：（需要放在 csv_dir_str 里）
for [train, val, test]:
distribution_xx_train.pkl  # 向量输出（x4 代表卷积层输出，xc 代表全连接层输出）
paths_train.pkl  # 数据路径
targets_train.pkl  # 数据真实标签

最终输出文件：（会放在 archive_dir 里）
folder
    |
    |—— DBLP-Scholar  # 沿用强哥单边决策树的旧输入文件名，内容用于稍后生成规则
    |       |—— 325
    |       |    |—— train.csv
    |       |—— pair_info_more.csv
    |
    |—— risk_dataset  # 用于给天伟风险分析的输入
    |       |—— all_data_info.csv
    |       |—— train.csv
    |       |—— val.csv
    |       |—— test.csv
    |
    |—— softmax  # 存放各神经网络全连接层的输出（即分类概率），用于 ensemble
    |       |—— r50
    |            |—— distribution_xc_train.csv
    |            |—— distribution_xc_val.csv
    |            |—— distribution_xc_test.csv
    |
    |—— decision_tree_rules_clean.txt  # 运行单边决策树后，才会生成的规则文件


【运行代码】
python get_risk_dataset.py
(然后去 ER_tree_2_rules 修改设置、运行代码)