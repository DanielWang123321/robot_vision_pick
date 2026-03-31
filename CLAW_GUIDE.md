# Vision Pick - Openclaw 集成指南

## 重要提示

- 项目对你只读，不要修改、删除、新增任何文件。
- 你与用户的交互要保持简洁
  - 告知用户桌面有什么，让用户选择抓取什么，
  - 每次抓取完成后告知执行结果。

## 项目概述

桌面水果抓取系统。xArm 6 机械臂 + RealSense D435 深度相机（eye-in-hand）+ 云端 LLM 视觉识别。
当前识别目标：**绿葡萄（green_grape_bunch）** 和 **猕猴桃（brown_kiwi）**。

## 环境准备

conda activate py312

## 运行方式

Openclaw 应使用**交互模式**，通过 stdin/stdout 与脚本交互：

```bash
python main.py --interactive
```

| 参数 | 作用 |
|------|------|
| `--interactive` | 交互模式：扫描后等待用户选择 |
| `-v` | 显示 INFO 日志；`-vv` 显示 DEBUG |
| `--dry-run` | 只计算坐标，不动机械臂 |
| `--loop` | 循环自动抓取（无需交互） |

无参数运行 (`python main.py`) = 自动抓取评分最高的目标。

## 交互模式流程

### 1. 扫描阶段
系统自动：移动到高位 (z=400mm) → 拍照 → LLM 识别 → 过滤 → 输出检测结果。

关键字段说明：
- `[T1]`/`[T2]`：目标跟踪 ID（跨扫描保持一致）
- `score`：多维度综合评分（置信度 + 可抓取性 + 边缘距离 + 深度等）
- `confidence`：LLM 识别置信度 (0-1)
- `graspable`：LLM 判断是否可抓取
- `hint`：LLM 给出的抓取建议

### 2. 选择阶段
系统等待 stdin 输入，接受以下格式：
- **数字**：`1` 或 `2`（列表序号）
- **目标 ID**：`T1` 或 `t1`（不区分大小写）
- **退出**：`0`、`q`、`quit`、`exit`

### 3. 执行阶段
选择后系统自动执行完整抓取流程：
1. 近距检测 + 视觉伺服居中（颜色匹配迭代校正）
2. 检测物体朝向 → 计算最优夹爪角度
3. 渐进下降 → 夹取 → 抬升 → 移动到释放位置 → 释放
4. **放置后验证**：回到拍照位重新扫描，对比前后检测结果

### 4. 结果反馈
执行后输出以下之一：
- `Picked successfully. Total: N` — 成功，经过验证确认物体已消失
- `Failed to pick <name>: <reason>` — 失败

常见 reason 值：

| reason | 含义 |
|--------|------|
| `success` | 抓取成功 |
| `empty_grasp` | 夹爪闭合但未夹到物体 |
| `verify_still_present` | 放置后验证发现物体仍在桌面（夹取失败） |
| `out_of_bounds` | 目标超出安全范围 |
| `xy_estimation_failed` | 坐标转换失败 |
| `identity_mismatch` | 近距验证发现不是预期物体 |

### 5. 循环
执行完一次后，系统自动进入下一轮扫描，重复上述流程。输入 `0` 退出。

退出时输出：`Session total: N`

## 异常情况

- **"No objects detected. Table is clear."** — 桌面无目标物体，程序退出
- **Ctrl+C** — 安全停止，机械臂回到初始位置
- **连续 Ctrl+C 两次** — 强制退出

## 诊断数据

每次运行自动保存到 `diagnostics/session_<timestamp>/`：
- `scans/scan_NNNN/color.jpg` — 扫描照片
- `scans/scan_NNNN/scan.json` — LLM 检测结果
- `picks/pick_NNNN/pick.json` — 抓取执行结果（含 verified 字段）
- `grasp_log.csv` — 所有抓取记录汇总

## 硬件参数（供参考）

- 机械臂 IP：192.168.1.60
- 安全范围：X[200,600] Y[-500,500] Z[0,500] (mm)
- 拍照位置：(300, -200, 400)
- 释放位置：(300, 350, 300)
- 夹爪：G2，最大速度 225mm/s，最大力 100
