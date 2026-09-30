abstract: 拿到一台新服务器的一些环境配置和一些linux指令，老是搞忘咋办呢
time: 2026/02/06
## 查看配置

看当前系统什么架构的，比如我是x86

> uname -m

## 装机

### SSH

* 值得一提的是我们熟悉的mobaxterm 是只支持windows不支持linux 的， 所以要ssh 的话只能用final shell。

查看本地是否已有 SSH 密钥

> ls ~/.ssh

如果你发现没有 SSH key，可以这样生成

> ssh-keygen -t ed25519 -C "your_email@example.com"

一路回车即可

查看公钥

> cat ~/.ssh/id_ed25519.pub

输出类似

ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAI... your_email@example.com
就是你的ssh key

然后到我们需要免密连接的服务器上粘贴公钥

> mkdir -p ~/.ssh
> chmod 700 ~/.ssh

然后编辑

> vim ~/.ssh/authorized_keys

把你刚复制的公钥粘进去，每个 key 占一行，保存。

vscode 配置

Ctrl + Shift + P

输入：`Remote-SSH: Open SSH Configuration File`

```
Host myserver
     HostName 1.2.3.4
     User ubuntu
     Port 22
     IdentityFile ~/.ssh/id_ed25519
```

### Anaconda

定位到个人文件夹

> cd ~

通过镜像安装miniconda

```bash
wget https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-latest-Linux-x86_64.sh
```

![](https://i-blog.csdnimg.cn/direct/c7cacbfd072549e7a59835c75a014699.png)

然后你的根目录就会有如下安装包，安装安装包

> bash Miniconda3-latest-Linux-x86_64.sh

一直回车或者yes, 你根目录就有个miniconda3 了

更多anaconda 细节

[https://blog.csdn.net/Scabbards_/article/details/128709999](https://blog.csdn.net/Scabbards_/article/details/128709999)

### Docker

指令参考

[https://blog.csdn.net/Scabbards_/article/details/148842535](https://blog.csdn.net/Scabbards_/article/details/148842535)

### Pycharm

我们下下来的是一个tar.gz 文件，需要解压，这个解压后就相当于安装了

> tar -xzf pycharm-*.tar.gz

进到 `bin` 目录，运行启动脚本：

> cd pycharm-*/bin
> ./pycharm.sh

### 其他软件

安装包形式

Ubuntu / Debian: .deb

CentOS / RHEL / Rocky / Fedora: .rpm

安装

sudo dpkg -i xxx.deb
sudo apt-get install -f

## 指令

### 重启

> sudo reboot

### **快捷键**

| **复制** | `Ctrl + Shift + C` |
| --- | --- |
| **粘贴** | `Ctrl + Shift + V` |

### tmux后台跑程序

新建session

```bash
tmux new -s train

# 开跑
python train.py
```

`Ctrl + B` 再按 `D` → 退出 tmux session

```bash
# 回来看跑的咋样了

tmux attach -t train
```

查看所有session

> tmux ls

删除session

> tmux kill-session -t train

## 清理

### 磁盘

查看磁盘

> df -h

如果满了，先清缓存：

> sudo apt clean

### 内存

查看内存

> top

### VDI 相关

去掉动画

> gsettings set org.gnome.desktop.interface enable-animations false

天哪去掉动画之后丝滑很多！

一件清理脚本

```bash
#!/bin/bash

echo "=============================="
echo "🧹 Safe Deep Clean Starting..."
echo "=============================="

# 1. pip cache
echo "➡️ Cleaning pip cache..."
pip cache purge 2>/dev/null

# 2. conda cache
echo "➡️ Cleaning conda cache..."
conda clean -a -y 2>/dev/null

# 3. general cache
echo "➡️ Cleaning ~/.cache ..."
rm -rf ~/.cache/* 2>/dev/null

# 4. GPU / profiling tools
echo "➡️ Cleaning NVIDIA / Nsight..."
rm -rf ~/nvidia 2>/dev/null
rm -rf ~/nsight-systems-* 2>/dev/null

# 5. temp downloads
echo "➡️ Cleaning fsdownload..."
rm -rf ~/fsdownload/* 2>/dev/null

# 6. python bytecode
echo "➡️ Cleaning __pycache__ ..."
find ~/.local -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null

# 7. pip local leftovers
echo "➡️ Cleaning pip leftovers..."
rm -rf ~/.local/share/pip 2>/dev/null

# 8. optional: journal logs shrink
echo "➡️ Cleaning system logs..."
sudo journalctl --vacuum-time=7d >/dev/null 2>&1

echo "=============================="
echo "✅ Cleaning Done!"
echo "=============================="

df -h
```
