abstract: 简单记录一下docker 部署vllm 服务， 为什么呢，因为我就是记不住啊
time: 2026/03/16
![](https://i-blog.csdnimg.cn/direct/edb7c24127d24e48bfd101d940b2a2e2.png)

## 下载

拿qwen-3.5-9B 为例：

[https://www.modelscope.cn/models/Qwen/Qwen3.5-9B](https://www.modelscope.cn/models/Qwen/Qwen3.5-9B)

首先下模型

```bash
git lfs install
git clone https://www.modelscope.cn/Qwen/Qwen3.5-9B.git
```

然后下对应docker,框架我用的是vllm

```bash
docker pull vllm/vllm-openai:qwen3_5
```

（官方文档: [https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen3.5.html#nvidia](https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen3.5.html#nvidia)）

docker images 检查一下下好了没，下好了就准备启动了↓

## 启动

因为docker 有启动命令，然后可以一键把模型部署上去运行

拆解一下启动命令

```bash
docker run -ti -d \
--privileged \
--gpus all -e CUDA_VISIBLE_DEVICES=0,1,2,3 \
--name qwen35-27b -v /data/home/yy/models/Qwen3.5-27B:/models -p 8005:8000 \
--ipc=host \
vult2.zhixuncloud.cn:8443/vllm/vllm-openai:qwen3_5 \
--model /models --served-model-name Qwen3.5-27B \
--tensor-parallel-size 4 \
--reasoning-parser qwen3 \
--enable-prefix-caching \
```

### Docker 运行参数

> docker run

启动一个容器。

> -ti

两个参数组合：

-t = 分配一个 tty终端

-i = 保持 stdin

主要是为了方便 docker attach 调试。

实际推理服务 可以不需要。

> -d

后台运行 container（daemon）。

否则终端会被占住。

> --privileged

给容器 最高权限。

通常为了：

GPU driver 访问/shared memory/CUDA runtime

很多 AI 容器都这么开。

> --gpus all

让 docker 把所有 GPU 挂到容器里。

控制用哪张卡：

```bash
CUDA_VISIBLE_DEVICES
-e CUDA_VISIBLE_DEVICES=0,1,2,3
```

容器名字：

> --name qwen35-9b

docker logs qwen35-9b
docker stop qwen35-9b

都靠它。

挂载模型目录

```bash
-v /data/home/yy/models/Qwen3.5-27B:/models
```

端口映射

```bash
-p 8005:8000
```

宿主机:8005  ->  容器:8000

访问 API：

http://server:8005/v1/chat/completions

```bash
--ipc=host
```

共享 host 的 shared memory。

原因：LLM 推理需要大量 KV cache shared memory。

镜像

```bash
vult2.zhixuncloud.cn:8443/vllm/vllm-openai:qwen3_5
```

### vllm 传参

> --model /models

指定模型路径。

因为你刚才挂载了：

-v /data/home/yy/models/Qwen3.5-27B:/models

所以这里读取：

```
/models/config.json
/models/tokenizer.json
/models/model.safetensors
```

> --served-model-name Qwen3.5-27B

API 返回的 model name。

例如请求：

```json
{
  "model":"Qwen3.5-27B"
}
```

> --tensor-parallel-size 4

模型 tensor parallel。

意思是：27B / 4 GPU。每张卡只存一部分权重。

> --enable-prefix-caching

vLLM 的 **核心优化之一**。KV cache **复用**。

> --reasoning-parser qwen3

这是 Qwen reasoning 模型专用。

作用：解析模型的 reasoning tokens：

> --gpu-memory-utilization 0.95

用多少现存

> --max-model-len 32768

最大 context。

> --api-key sk-76db750bbxxxxxxxxxxxxxxxxx

api key, 密码配置

## 部署检查

### **models 接口**

```bash
curl http://localhost:8000/v1/models
```

返回类似

```json
{
  "data":[
    {
      "id":"qwen2.5-7b",
      "object":"model"
    }
  ]
}
```

就说明api server 正常

### 调用推理

curl 测试

写个bash 脚本，然后bash 一下试试

```bash
curl http://localhost:8000/v1/chat/completions \
 -H "Content-Type: application/json" \
 -d '{
  "model":"qwen",
  "messages":[
    {"role":"user","content":"你好"}
  ]
 }'
```

然后直接看返回就行

### python 调用

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="EMPTY"
)
resp = client.chat.completions.create(
    model="qwen",
    messages=[{"role": "user", "content": "hello"}]
)
print(resp.choices[0].message.content)
```

## 运行健康检查

呃呃，公司服务器被挖矿了，所以这就体现出运行健康检查的重要性了

不过防挖矿是运维同事的事情了，所以我们主要检查运行健康

同时我mentor 说官方的健康接口只能看server 是否存活，但是有的时候虽然server 依旧存活，但是模型的推理还是会出问题的

所以我们的健康检查脚本

1. 检查服务是否存活

2. 小量调用推理，检查推理是否健在

```bash
#!/bin/bash
# vLLM 持续健康检查脚本（带日志）

set -euo pipefail

MODEL_NAME="Qwen3.5-27B"
ENDPOINT="http://localhost:8005"
TIMEOUT=10
API_KEY="sk-76dxxxxxxxxx"
AUTH_HEADER="Authorization: Bearer ${API_KEY}"

# 日志文件
LOG_FILE="/var/log/vllm_healthcheck.log"

# 检查间隔（秒）
INTERVAL=30

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "${LOG_FILE}"
}

log "=== vLLM Health Check Daemon Started ==="

while true; do
    log "Starting health check..."

    # 1. /health
    if ! curl -f -s --max-time "${TIMEOUT}" "${ENDPOINT}/health" > /dev/null; then
        log "[ERROR] /health endpoint failed"
        sleep "${INTERVAL}"
        continue
    fi
    log "[OK] /health"

    # 2. /v1/models
    MODEL_RESPONSE=$(curl -f -s --max-time "${TIMEOUT}" -H "${AUTH_HEADER}" "${ENDPOINT}/v1/models" 2>&1) || {
        log "[ERROR] /v1/models failed"
        log "Response: ${MODEL_RESPONSE}"
        sleep "${INTERVAL}"
        continue
    }
    log "[OK] /v1/models"

    # 3. 推理测试
    RESPONSE=$(curl -f -s --max-time "${TIMEOUT}" -X POST \
        "${ENDPOINT}/v1/completions" \
        -H "Content-Type: application/json" \
        -H "${AUTH_HEADER}" \
        -d '{
            "model": "'"${MODEL_NAME}"'",
            "prompt": "healthcheck",
            "max_tokens": 1,
            "temperature": 0,
            "stream": false
        }' 2>&1) || {
        log "[ERROR] Inference request failed"
        log "Response: ${RESPONSE}"
        sleep "${INTERVAL}"
        continue
    }

    if [[ -n "${RESPONSE}" && "${RESPONSE}" != *"error"* ]]; then
        log "[OK] Inference test passed"
    else
        log "[ERROR] Inference test failed"
        log "Response: ${RESPONSE}"
    fi

    log "Health check completed"
    log "----------------------------------------"

    sleep "${INTERVAL}"
done
```

然后后台运行

> nohup bash vllm_healthcheck_daemon.sh > /dev/null 2>&1 &

当然也得支持打印日志：

> tail -f /var/log/vllm_healthcheck.log
