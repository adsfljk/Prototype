import numpy as np

# 解析规则函数
def parse_rules(file_path):
    rules = []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip().replace('!=', '!=').replace('=', '==')
            if 'then' in line:
                conditions, label = line.split(' then ')
                label = int(label.strip())
                conditions = conditions.replace(' if ', '').split(' if ')
                rules.append((conditions, label))
    return rules

# 检查样本是否满足规则条件
def match_rule(conditions, sample):
    for condition in conditions:
        feature, operation, value = condition.split()
        feature = int(feature.replace('feature_', ''))
        value = float(value)
        if not eval(f'sample[{feature}] {operation} {value}'):
            return False
    return True

# 根据规则进行预测
def predict(rules, X):
    predictions = []
    for sample in X:
        predicted = None
        for conditions, label in rules:
            if match_rule(conditions, sample):
                predicted = label
                break
        if predicted is None:
            predicted = -1  # 如果没有匹配到任何规则，返回默认值
        predictions.append(predicted)
    return predictions

# 计算准确率
def calculate_accuracy(y_true, y_pred):
    correct = np.sum(np.array(y_true) == np.array(y_pred))
    return correct / len(y_true)

# 主函数
def main():
    # 读取规则
    rules = parse_rules('rules.txt')

    # 假设已经加载了X_train和y_train
    X_train = np.load('X_train.npy')  # X_train: 特征矩阵，二进制特征
    y_train = np.load('y_train.npy')  # y_train: 标签

    # 预测
    y_pred = predict(rules, X_train)

    # 计算准确率
    accuracy = calculate_accuracy(y_train, y_pred)
    print(f'Accuracy: {accuracy:.4f}')

if __name__ == '__main__':
    main()
