# 使用os库方法示例
import os

path = r"D:\AI_Graph\sam2output\3.mp4"

dir_path = os.path.dirname(path)
file_name = os.path.basename(path)
name, ext = os.path.splitext(file_name)

print(f"dirname: {dir_path}")
print(f"basename: {file_name}")
print(f"splitext: {name}, {ext}")

A = "dirname"
B = "filename"
AjointB = os.path.join(A, B)

print(f"AjointB: {AjointB}")
