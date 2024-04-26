## 新建一个 anaconda 环境，用 python=3.9

```shell
conda create --name cardi python=3.9
```

## 激活环境

```shell
conda activate cardi
```

---

# 安装库

## 安装依赖

```shell
pip install -r requirements.txt
```

## 安装 cuda 版 torch（如有 NVIDIA GPU）

参考 https://pytorch.org/get-started/pytorch-2.0/ 进行安装。

## 安装其他依赖

```shell

# 进入/FACE/torchquadMy目录，安装重写的torchquad
cd ./FACE/torchquadMy
pip install .
```
