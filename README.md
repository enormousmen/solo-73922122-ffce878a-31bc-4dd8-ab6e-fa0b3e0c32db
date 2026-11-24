# vision-edge

产线相机缺陷分类边缘服务：FastAPI 接收 3 路固定相机的 64×64 灰度图，
CPU 上跑本地 ONNX 分类模型，值班台可随时查询每路最近一次算完的结果。

## 语义（与需求逐条对应）

| 需求 | 实现 |
| --- | --- |
| 每路最多留一张待算图 | 每路一个 `pending` 槽位（`app/hub.py`） |
| 新帧顶替待算旧帧时沿用原排位 | 全局 FIFO 队列存的是**路名**，顶替只换数据、不动队列位置 |
| 帧号未超过本路已接收最大值就拒收 | 每路记录 `max_received`，`frame_no <= max_received` → `409` |
| 两个推理通道共用一份已加载的 CPU 模型，最多同时算两批 | 启动时加载**一个** `OnnxClassifier`，`INFER_CHANNELS=2` 个 worker 协程共用；Hub 用 `_active_batches` 闸门保证在批批数 ≤ 通道数 |
| 有空闲通道就照原规则从队首取批 | `take_batch()` 在条件锁内原子弹出队首最多 `BATCH_LANES` 路；通道忙时在锁内等待，批结束 `notify_all` 唤醒 |
| 同一路还有帧在算也可再次入批 | 出队只清 `queued`、不动 `in_flight`；新帧到达重新排队尾，可被另一通道取走（同一路最多 2 帧在算） |
| 成功帧只要比该路已发布的帧更新，算完即可查，无需等早批 | `finish_batch()` 仅当 `frame_no > last_result.frame_no` 才发布；晚到的旧帧计入 `stale_frames`，不覆盖新结果 |
| 新帧失败保留原结果；早先入批的旧帧仍按同规则发布 | `fail_batch()` 不动 `last_result`，只释放该批的在算计数；另一通道里的旧帧结批时照常走发布规则 |
| 各路还有帧在算时 `in_batch=true`，最后一帧结批后回 `false` | `Lane.in_batch` 是 `in_flight > 0` 的派生属性；每帧结批（成功/失败）减一 |
| 推理期间照常收图和查询 | ONNX 推理放线程池执行，事件循环不阻塞；两批可并行在两个线程上跑 |
| 查询返回上次成功的帧号和分类 | `GET /cameras/{lane}/result`（接口不变） |
| 收帧 API 不变 | `POST /cameras/{lane}/frames?frame_no=N`，仅新增可选演示请求头 |

## 启动（Docker Compose）

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
#      "in_batch":false,"in_flight":0,"queue_position":0,"queue":["cam1"]}

# 帧号不递增 → 拒收
curl -i -X POST 'http://localhost:8000/cameras/cam1/frames?frame_no=1' \
     -H 'Content-Type: application/octet-stream' --data-binary @/tmp/f1.raw
# 409 {"detail":"frame_no 1 <= max received 1 on cam1"}

# 可选演示头：让包含该帧的批次在 CPU 上额外睡 N 毫秒（0..10000，默认用 INFER_DELAY_MS）
curl -i -X POST 'http://localhost:8000/cameras/cam1/frames?frame_no=2' \
     -H 'Content-Type: application/octet-stream' \
     -H 'X-Infer-Delay-Ms: 1500' --data-binary @/tmp/f1.raw

# 查询该路上次成功结果（帧号是实际算过且最新的那一帧，无需等更早的慢批）
curl -s http://localhost:8000/cameras/cam1/result
# {"lane":"cam1","frame_no":1,"class_id":0,"scores":[...],"computed_at":...}

# 队列/车道总览（含 channels、active_batches、各路 in_flight/stale_frames）
curl -s http://localhost:8000/cameras
```

## 演示

### 1. 同一路的新帧先出结果（双通道 + 可控延迟）

```bash
python3 demo/demo_channels.py        # 仅需标准库；默认打 http://localhost:8000
```

脚本流程（确定性，不依赖机器快慢）：

1. cam1 送帧 N（`X-Infer-Delay-Ms: 1500`），通道 1 开始算慢批；
2. 慢批未结束时再送 cam1 帧 N+1（延迟 100ms），该路重新入队、被空闲的通道 2 取走，
   快照里同一路 `in_flight=2`、`in_batch=true`；
3. 轮询结果接口：帧 N+1 约 0.1s 后即可查（不用等 1.5s 的慢批）；
4. 慢批结束后结果仍是帧 N+1，旧帧计入 `stale_frames`，`in_batch` 回到 `false`；
5. 再演示慢批占着通道 1 时，另一路 cam2 的快帧由通道 2 即时算出。

### 2. 一路频繁换图，另一路仍按排位算出（原演示，保留）

```bash
python3 demo/demo.py
```

cam1/cam2 各送一帧 → cam0 连续冲 60 帧（每帧顶替上一帧待算图）→
最终 cam1/cam2 的结果帧号就是自己送的那帧，cam0 的结果是取批时实际待算的那一帧。

## 本地开发与测试

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements-dev.txt
.venv/bin/python scripts/make_model.py --smoke-test   # 重新生成 ONNX 模型
.venv/bin/python -m pytest tests/ -q                  # 25 个测试
.venv/bin/python -m uvicorn app.main:app --port 8000  # 本地起服务
```

## 配置（环境变量）

| 变量 | 默认 | 说明 |
| --- | --- | --- |
| `LANES` | `cam0,cam1,cam2` | 固定相机路名 |
| `BATCH_LANES` | `2` | 每批最多取几路 |
| `INFER_CHANNELS` | `2` | 推理通道数（并发批次数上限），所有通道共用一个模型会话 |
| `MODEL_PATH` | `app/model/classifier.onnx` | ONNX 模型路径 |
| `INFER_DELAY_MS` | `0` | 演示用：模拟单批推理耗时（compose 里为 200） |
| `MAX_HINT_DELAY_MS` | `10000` | `X-Infer-Delay-Ms` 请求头允许的最大值 |

## 模型

`app/model/classifier.onnx`：`Flatten → Gemm(4096→3)`，输入
`float32[batch,1,64,64]`（动态 batch），输出 3 类 logits，服务内做 softmax。
由 `scripts/make_model.py` 生成（固定随机种子，可复现）；换成真实缺陷模型时
保持输入/输出名（`input`/`logits`）或同步修改 `app/model.py`。

## 剩余事项

- 结果为内存态，重启即清空；需要持久化可挂 Redis/SQLite。
- 批次失败当前直接丢弃该批帧（保留原结果），如需重试可在 `fail_batch`
  里把帧重新入队（注意避免无限重试）。
- 通道数默认 2；真实重模型下并发批次数受核数限制，可用 `INFER_CHANNELS` 调。
- 相机鉴权、TLS、限流未实现，按部署环境补。
- 演示用 `INFER_DELAY_MS` 与 `X-Infer-Delay-Ms` 只是观测开关，
  生产环境应关闭（去掉请求头并把延迟设为 0）。
