import re
import os

def del_rule(model_name,dataset,log_class_name ,last_modified_delete_content):


    root = "/home/zhuyijia/prototypical/V5_DT_ToN-IoT/entity/"+str(model_name)+"/"+str(dataset)+'/'

    file_name = log_class_name+"_setup.py"


    # 读取文件
    with open(os.path.join(root, file_name), 'r') as file:
        content = file.read()

    # 定义需要删除的参数列表
    params_to_remove = ["code","code0", "code1", "code2", "code3", "code4", "port", "cls"]

    # 正则表达式：匹配以 add_with_ac 开头的函数并提取对象、函数名和参数
    pattern = r'(\w+)\.add_with_ac_\w+\((.*?)\)'

    def replace_function(match):
        # 提取对象名称和参数
        obj, args = match.groups()
        
        # 分割参数列表并去除空白符
        args_list = [arg.strip() for arg in args.split(",")]

        # 过滤掉需要删除的参数
        filtered_args = []
        for arg in args_list:
            key_value = arg.split("=")
            if len(key_value) == 2 and key_value[0].strip() not in params_to_remove:
                filtered_args.append(arg)
        
        # 重新组合剩余的参数
        new_args = ", ".join(filtered_args)
        
        # 返回新的函数调用形式，替换为 ddelete
        return f"{obj}.delete({new_args})"

    # 使用正则表达式替换符合条件的函数调用
    modified_delete_content = re.sub(pattern, replace_function, content)


    # 正则表达式：匹配并删除整个 clear_all 函数的部分，包括其调用
    pattern_clear_all = r'def clear_all\([^\)]*\):.*?clear_all\([^\)]*\)'
    # 删除 clear_all 函数部分
    modified_delete_content = re.sub(pattern_clear_all, '', modified_delete_content, flags=re.DOTALL)

    # 删除文件中的 "bfrt.complete_operations()" 行
    modified_delete_content = re.sub(r'\s*bfrt\.complete_operations\(\)\s*', '', modified_delete_content)
    # 正则表达式匹配 "p4 = bfrt." 后面可能有不同的值，例如 netbeacon_ton_iot 等
    pattern_p4 = r'p4\s*=\s*bfrt\.\w+\.pipe'
    # 删除匹配的 "p4 = bfrt.<任意值>.pipe" 行
    modified_delete_content = re.sub(pattern_p4, '', modified_delete_content)
    
    


    # 正则表达式匹配 "p4 = bfrt.<任意值>.pipe" 后面插入 "delete()"
    pattern_p4 = r'(p4\s*=\s*bfrt\.\w+\.pipe)'

    if len(last_modified_delete_content) != 0:

        # 正则表达式：匹配并删除整个 clear_all 函数的部分，包括其调用
        pattern_clear_all = r'def clear_all\([^\)]*\):.*?clear_all\([^\)]*\)'
        # 删除 clear_all 函数部分
        content = re.sub(pattern_clear_all, '', content, flags=re.DOTALL)

    # 在每个匹配的行后加上 delete()
    modified_content = re.sub(pattern_p4, r'\1\n\n'+ last_modified_delete_content, content)


    # 将修改后的内容写回文件
    with open(os.path.join(root, file_name), 'w') as modified_file:
        modified_file.write(modified_content)
    
    return modified_delete_content
