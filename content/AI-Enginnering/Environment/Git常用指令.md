abstract: 记命令记不住，我外置大脑
time: 2023/10/12
参考视频[ 01 - Git - 教程简介_哔哩哔哩_bilibili](https://www.bilibili.com/video/BV1wm4y1z7Dg?p=1&vd_source=f04f16dd6fd058b8328c67a3e064abd5)

在需要使用git的文件夹打开git bash，指令如下↓

当然图形化界面也很香！github desktop也很舒服！

## 查看文件

版本号

> git cat-file -p + 版本号

## 仓库操作

![](https://i-blog.csdnimg.cn/blog_migrate/234f43c7b6390c31a575b815aa64a505.png)

### 在当前文件夹下克隆仓库

> git clone 仓库地址

### 配置文件

此处修改会在.git\config里面找到

> git config 配置名称 配置值

然后修改全部仓库的配置，会在c盘user .gitconfig里面找到

> git config --global 配置名称 配置值

### 设置用户

> git config user.name "你的名字"
> git config user.email "你的github邮箱"

## 文件操作

### 查看状态

> git status

### 添加文件到暂存区

> git add 文件名

### 添加所有相同后缀的文件到暂存区

> git add *文件名

### 从暂存区放回工作区

> git rm --cached 文件名

### 把暂存区的文件存贮到仓库中

（此处提交信息是自己写的备注）

> git commit -m 提交信息

### 查看历史操作

oneline是一行显示，更清楚一点

> git log --oneline

### 误删除

这个方法恢复误删除的工作区文件 ，记住是工作区！提交之后就恢复不了了哦！

> git restore 文件名

哼哼哼就算是提交了误删也没关系！直接把当前的版本库重置到某一次提交就行

> git reset --hard 版本号

但是如果不想删掉提交的话那就多加一个还原的版本

> git revert  版本号

### 回退

> git reset --hard HEAD

作用是：

- 丢弃 **所有未提交的修改**
- 工作区恢复到 **当前 commit 的状态**

如果还有新增的未跟踪文件（比如新建的文件）

> git clean -fd

(`reset --hard` **不会删除新文件)**

## 分支操作

### 创建新分支

分支在.git\refs\heads 里面查看哦

> git branch 新分支名

### 查看分支

> git branch -v

带星号的是正在使用的分支

### 切换分支

> git checkout 要切换的分支名

### 创建并切换分支

> git checkout -b 创建并切换的分支名

### 删除分支

> git branch -d 要删除的分支名

（居然没有确认操作欸）

### 合并

> git merge 要合并的分支名

### 冲突

万一冲突了就点开冲突的文件看比对的内容，直接修改文件然后add和commit

## 标签操作

### 添加标签

给当前的提交版本设置一个别名，这样就不用每次一长串版本号了

> git tag  标签名  版本号

注意，可以给每一个版本增加一个标签但是标签不能重复

### 删除标签

> git tag -d 被删除的标签名

### 根据标签创建分支

> git checkout -b 标签名（被创建的分支名）

## 远程仓库

远程仓库配置在.git config上

![](https://i-blog.csdnimg.cn/blog_migrate/82d86aecf247db06b882c9f88a4ef858.png)

### ssh

作为科研人仓库经常在论文出来之前不会公开，所以ssh 是个常用的处理私有仓库的办法

1） 生成ssh-key

首先我们在服务器上生成我们的github 私钥

> ssh-keygen -t ed25519 -C "github"

一直回车，这一步不太可能出错

然后再输入

> cat ~/.ssh/id_ed25519.pub

就能得到一串类似ssh-ed25519 AAAAC3Nz... 的内容

2)  添加到github

Settings
 → SSH and GPG keys
 → New SSH key

填写服务器名和刚刚得到的ssh-ed25519 AAAAC3...就行

测试github 连接

> ssh -T git@github.com

出现 Hi Scabbards1500! You've successfully authenticated, but GitHub does not provide shell access. 就成功了

3）克隆仓库

> git clone git@github.com:用户名/仓库.git

### 把本地仓库内容推送到远程仓库

一般方式

这里就用它一般默认的名称origin演示了

> git push origin

注意，这里可能会出现问题

fatal: unable to access 'https://github.com/.../.git':Recv failure Connection was rese/

fatal: unable to access 'https://github.com/Scabbards1500/INSTANCE2022-VLSeg.git/': Failed to connect to github.com port 443 after 21072 ms: Timed out

解决方式：

配置咳咳, 见后文

### 从远程仓库拉取改动

> git pull origin

## 咳咳

### 配置

```bash
git config --global http.proxy http://username:password@proxy.example.com:port
git config --global https.proxy https://username:password@proxy.example.com:port
```

### 查看xx

> git config --get http.proxy
> git config --get https.proxy

### 删除xx

> git config --global --unset http.proxy
> git config --global --unset https.proxy

## 从github获取仓库并修改仓库内容

工具 gitbash

**1. 克隆 GitHub 仓库到本地**：

```bash
git clone https://github.com/yourusername/repository.git
```

2.修改，添加

> git add .

3.使用 `git commit` 命令提交暂存区中的更改并添加提交消息

> git commit -m "Your commit message here"

4.最后，使用 `git push` 命令将更改推送到 GitHub 仓库：

> git push origin master(分支)
