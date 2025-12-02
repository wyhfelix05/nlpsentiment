import subprocess
import re

# 输出文件
output_file = "requirements.txt"

# 获取所有包
result = subprocess.run(["pip", "freeze"], capture_output=True, text=True, check=True)
packages = result.stdout.splitlines()

clean_packages = []
seen = set()

for pkg in packages:
    pkg = pkg.strip()
    if not pkg or pkg.startswith("#"):
        continue
    # 过滤掉本地路径安装的包（@ file:///...）
    if " @ file://" in pkg:
        # 尝试保留包名和版本号
        match = re.match(r"([a-zA-Z0-9_\-]+)==([\d\.]+)", pkg)
        if match:
            pkg = f"{match.group(1)}=={match.group(2)}"
        else:
            continue
    # 去掉重复包
    if pkg in seen:
        continue
    seen.add(pkg)
    clean_packages.append(pkg)

# 写入文件
with open(output_file, "w", encoding="utf-8") as f:
    for line in sorted(clean_packages):
        f.write(line + "\n")

print(f"requirements.txt 已生成，共 {len(clean_packages)} 个包。")
