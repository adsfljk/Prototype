# 示例优先级字典
priority_dict = {
    "10001": 2,
    "20005": 1,
    "10000": 12,
    "10005": 2,
}

# 排序：先按value升序，再按key升序
sorted_items = sorted(priority_dict.items(), key=lambda x: (x[1], int(x[0])))

# 将优先级替换为排序后的排位序号+1
ranked_dict = {item[0]: index + 1 for index, item in enumerate(sorted_items)}

# 打印结果
for key, value in ranked_dict.items():
    print(f"序号: {key}, 排位: {value}")
