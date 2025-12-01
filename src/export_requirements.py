import subprocess

# 输出文件
output_file = "requirements.txt"

# 调用 pip freeze 获取所有包
result = subprocess.run(["pip", "freeze"], capture_output=True, text=True, check=True)
packages = result.stdout.splitlines()

# 写入文件
with open(output_file, "w", encoding="utf-8") as f:
    for line in packages:
        f.write(line + "\n")

# 输出信息到终端
print(f"requirements.txt 已生成，共 {len(packages)} 个包。")
print("包列表如下：")
for line in packages:
    print(line)
