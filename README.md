# vision-edge

产线相机缺陷分类边缘服务：FastAPI 接收 3 路固定相机的 64×64 灰度图，
CPU 上跑本地 ONNX 分类模型，值班台可随时查询每路最近一次算完的结果。

## 语义（与需求逐条对应）

| 需求 | 实现 |
| --- | --- |
| 每路最多留一张待算图 | 每路一个 `pending` 槽位（`app/hub.py`） |
| 新帧顶替待算旧帧时沿用原排位 | 全局 FIFO 队列存的是**路名**，顶替只换数据、不动队列位置 |
| 帧号未超过本路已接收最大值就拒收 | 每路记录 `max_received`，`frame_no <= max_received` → `409` |
| 单推理通道每批取最前两路各一张，只有一路就算一张 | 单个 `inference_loop` 协程，`take_batch()` 从队首弹出最多 2 路 |
| 取走后该路新图重新排队 | 出队即清 `queued` 标记，新帧到达重新入队尾 |
| 推理期间照常收图和查询 | ONNX 推理放线程池执行，事件循环不阻塞 |
| 已入批的图继续算完，结果带实际算过的帧号 | 结果取自出队时快照的 `PendingFrame.frame_no` |
| 查询返回上次成功的帧号和分类 | `GET /cameras/{lane}/result` |
| 批次失败保留原结果，继续处理待算图 | 异常 → `fail_batch()`（不动 `last_result`），循环继续 |

## 启动

```bash
docker compose up --build
# 服务监听 8000 端口；compose 里设了 INFER_DELAY_MS=200 模拟更重的模型，
# 让排队/顶替行为可观测，生产可设为 0。
```

## API 示例

```bash
# 送一帧（body 为 4096 字节原始灰度图，行优先，无头）
head -c 4096 /dev/urandom > /tmp/f1.raw
curl -i -X POST 'http://localhost:8000/cameras/cam1/frames?frame_no=1' \
     -H 'Content-Type: application/octet-stream' --data-binary @/tmp/f1.raw
# 202 {"lane":"cam1","frame_no":1,"accepted":true,"replaced":false,
#      "in_batch":false,"queue_position":0,"queue":["cam1"]}

# 帧号不递增 → 拒收
curl -i -X POST 'http://localhost:8000/cameras/cam1/frames?frame_no=1' \
     -H 'Content-Type: application/octet-stream' --data-binary @/tmp/f1.raw
# 409 {"detail":"frame_no 1 <= max received 1 on cam1"}

# 查询该路上次成功结果（帧号是实际算过的那一帧）
curl -s http://localhost:8000/cameras/cam1/result
# {"lane":"cam1","frame_no":1,"class_id":0,"scores":[...],"computed_at":...}

# 队列/车道总览
curl -s http://localhost:8000/cameras
```

## 演示：一路频繁换图，另一路仍按排位算出

```bash
python3 demo/demo.py            # 仅需标准库；默认打 http://localhost:8000
```

脚本流程：cam1/cam2 各送一帧 → cam0 连续冲 60 帧（每帧顶替上一帧待算图）
→ 中途快照可见 cam0 在队尾、cam1 已入批继续算 → 最终 cam1/cam2 的结果
帧号就是自己送的那帧，cam0 的结果是取批时实际待算的那一帧（`replaced_frames`
计数显示 59 帧被顶替）。实测输出见交付说明。

## 本地开发与测试

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements-dev.txt
.venv/bin/python scripts/make_model.py --smoke-test   # 重新生成 ONNX 模型
.venv/bin/python -m pytest tests/ -q                  # 17 个测试
.venv/bin/python -m uvicorn app.main:app --port 8000  # 本地起服务
```

## 配置（环境变量）

| 变量 | 默认 | 说明 |
| --- | --- | --- |
| `LANES` | `cam0,cam1,cam2` | 固定相机路名 |
| `BATCH_LANES` | `2` | 每批最多取几路 |
| `MODEL_PATH` | `app/model/classifier.onnx` | ONNX 模型路径 |
| `INFER_DELAY_MS` | `0` | 演示用：模拟单批推理耗时 |

## 模型

`app/model/classifier.onnx`：`Flatten → Gemm(4096→3)`，输入
`float32[batch,1,64,64]`（动态 batch），输出 3 类 logits，服务内做 softmax。
由 `scripts/make_model.py` 生成（固定随机种子，可复现）；换成真实缺陷模型时
保持输入/输出名（`input`/`logits`）或同步修改 `app/model.py`。

## 剩余事项

- 结果为内存态，重启即清空；需要持久化可挂 Redis/ SQLite。
- 批次失败当前直接丢弃该批帧（保留原结果），如需重试可在 `fail_batch`
  里把帧重新入队（注意避免无限重试）。
- 相机鉴权、TLS、限流未实现，按部署环境补。
- 演示用 `INFER_DELAY_MS` 只是观测开关，真实重模型下删除即可。
