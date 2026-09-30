abstract: 在jupyter notebook中导入现有环境(pytorch)
time: 2023/02/23
大家是否在jupyter notebook 里面只能找到默认的 Python3（ipykernel）呢？

![](https://i-blog.csdnimg.cn/blog_migrate/ad73f4d702a5e74d1d5e1cbb1d6ea760.png)

1.首先激活所需的环境

```
(base) C:\Users\scaaa>conda activate pytorch
```

2.如果需要导入现有环境，我们先要保证该环境支持jupyter notebook,有该功能,若没有则：

```
(pytorch) C:\Users\scaaa>pip install ipykernel
```

3.命名

```
(pytorch) C:\Users\scaaa>python -m ipykernel install --user --name "pytorch"
```

4.打开jupyter notebook，就可以看到该环境

```
(pytorch) C:\Users\scaaa>jupyter notebook
```

![](https://i-blog.csdnimg.cn/blog_migrate/e9fb632d6e55000d4e13514b079c18e3.png)
