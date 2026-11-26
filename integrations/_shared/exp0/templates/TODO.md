# {{GAME_NAME}} Exp 0 v0 待办

交付 40 个完整、可审计的回合，再用同一个真实诊断模型对同一批回合比较 `read` 和 `probe`。每局产出一个组件判断（`harness`、`weights`、`controller` 或弃权）。整局结果及其因果证据独立记录。v0 面向开发与复现，探测没有提升也可以交付。[{{ISSUE}}]({{ISSUE_URL}}) · [Exp 0 #174](https://github.com/HectorHHZ/llm-honing/issues/174)

勾选时附上对应的提交、命令输出、文件或发布回执。

## 1. 打通基线并冻结采集协议
- [ ] 固定游戏版本、harness 快照、controller、观察条件和两个 checkpoint，写入 `collection.json`。{{PINS_EVIDENCE}}
- [ ] 定义“一局”：起点断点、原生终止条件（win / loss / timeout / other）、上限、reset 组与配对方式。{{EPISODE_EVIDENCE}}
- [ ] 干净策略试跑与时间估计。{{PILOT_EVIDENCE}}

## 2. 构建和验证三类改动
- [ ] **Harness**：{{HARNESS_FAULT_RULE}} {{HARNESS_SMOKE}}
- [ ] **Weights**：整局由备用 checkpoint 服务；同一请求概率差 > 0.01。{{WEIGHTS_SMOKE}}
- [ ] **Controller**：{{CONTROLLER_FAULT_RULE}} {{CONTROLLER_SMOKE}}

## 3. 收集、标注和核验 40 局
- [ ] 40 局 10/10/10/10，全部尝试、排除原因、分组与切分。`validate_episode.py`、`audit_set.py` 输出。
- [ ] 本地包：`bundle.py prepare` 与 `bundle.py verify`。

## 4. 验证诊断输入和探测
- [ ] `diagnosis_inputs.py` 输入隔离；`probes.py selfcheck` 重渲染一致；`probes.py compare` 身份检查（0.01）。

## 5. 运行真实诊断并复算指标（需用户批准模型与账号）
- [ ] `read` 与 `probe` 各 40 次，保存回复、尝试、用量和耗时；`score.py` 离线复算。

## 6. 发布并提交验收回执（需用户批准）
- [ ] `bundle.py release` 与 `tools/verify_release.py`；上传与回执由用户决定。
