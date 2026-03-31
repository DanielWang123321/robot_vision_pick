# Vision Pick - OpenClaw MCP 集成指南

## 重要提示

- 项目对你只读，不要修改、删除、新增任何文件。
- 你与用户的交互要保持简洁：告知用户桌面有什么，让用户选择抓取什么，每次抓取完成后告知执行结果。

## 项目概述

桌面水果抓取系统。xArm 6 机械臂 + RealSense D435 深度相机（eye-in-hand）+ 云端 LLM 视觉识别。
当前识别目标：**绿葡萄（green_grape_bunch）** 和 **猕猴桃（brown_kiwi）**。

## 环境准备

```bash
conda activate xxx 
```

## MCP Server 配置

在 OpenClaw 的 MCP 配置中添加：

```json
{
  "mcpServers": {
    "vision-pick": {
      "command": "python",
      "args": ["mcp_server.py"],
      "cwd": "<项目绝对路径>",
      "env": {
        "VISION_PICK_DRY_RUN": "0"
      }
    }
  }
}
```

设置 `VISION_PICK_DRY_RUN=1` 可跳过真实机械臂运动（仅计算坐标）。

## MCP Tools

### `scan_table`

扫描桌面，返回检测到的物体列表（按抓取优先级排序）。

- **输入**：无参数
- **输出**：
```json
{
  "objects": [
    {
      "target_id": "T1",
      "name": "green_grape_bunch",
      "category": "fruit",
      "score": 8.5,
      "confidence": 0.92,
      "graspable": true,
      "hint": "grab from stem side"
    }
  ],
  "count": 1
}
```

字段说明：
- `target_id`：目标跟踪 ID（跨扫描保持一致）
- `score`：多维度综合评分（越高越优先）
- `confidence`：LLM 识别置信度 (0-1)
- `graspable`：LLM 判断是否可抓取
- `hint`：LLM 给出的抓取建议

### `pick_object`

抓取指定目标。必须先调用 `scan_table`。

- **输入**：`target_id`（字符串，如 `"T1"`）
- **输出**：
```json
{
  "success": true,
  "object_name": "green_grape_bunch",
  "reason": "success",
  "verified": true
}
```

| reason 值 | 含义 |
|-----------|------|
| `success` | 抓取成功 |
| `empty_grasp` | 夹爪闭合但未夹到物体 |
| `verify_still_present` | 放置后验证发现物体仍在桌面 |
| `out_of_bounds` | 目标超出安全范围 |
| `identity_mismatch` | 近距验证发现不是预期物体 |
| `no_scan` | 未先调用 scan_table |
| `target_not_found` | target_id 不存在 |

抓取后扫描缓存自动清空，需重新调用 `scan_table`。

### `get_status`

获取系统状态。

- **输入**：无参数
- **输出**：
```json
{
  "robot_connected": true,
  "dry_run": false,
  "total_picked": 3,
  "has_cached_scan": false
}
```

## 交互流程

1. 调用 `scan_table` 获取物体列表
2. 将结果展示给用户，让用户选择要抓取的目标
3. 调用 `pick_object(target_id="T1")` 执行抓取
4. 告知用户执行结果
5. 如需继续，回到步骤 1

## 异常情况

- `scan_table` 返回 `count: 0` — 桌面无目标物体
- MCP server 断开 — 检查机械臂和相机连接
