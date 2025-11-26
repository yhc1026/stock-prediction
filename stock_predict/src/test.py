import sys
import os

# 获取当前文件的绝对路径
current_file = __file__
# 获取当前文件所在目录
current_dir = os.path.dirname(current_file)
# 获取项目根目录
project_root = os.path.dirname(current_dir)

# 打印调试信息
print("当前文件:", current_file)
print("当前目录:", current_dir)
print("项目根目录:", project_root)

# 手动将config.py作为模块加载
config_path = os.path.join(project_root, 'config.py')
print("config.py路径:", config_path)

if os.path.exists(config_path):
    # 方法1：使用importlib
    import importlib.util
    spec = importlib.util.spec_from_file_location("config", config_path)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    print("✅ config加载成功!")
else:
    print("❌ config.py文件不存在")