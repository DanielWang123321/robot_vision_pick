# Vision Pick

基于 xArm 6 + RealSense D435 的桌面物体抓取系统，支持**云端 LLM 视觉**（Claude）和**本地 YOLO**两种检测后端。

## 硬件

- xArm 6 机械臂 (192.168.1.xx)
- Gripper G2 夹爪 (最大开口 84mm)
- Intel RealSense D435 深度相机 (eye-in-hand)

## 安装

```bash
conda activate py312
pip install -r requirements.txt
```

**使用 LLM 后端（默认）**，在 `.env` 中配置 API Key：
```
OPENROUTER_KEY=your_key_here
```

**使用 YOLO 后端**，无需 API Key，需先训练模型（见下方[YOLO 使用指南](#yolo-使用指南)）。

## 使用

```bash
# 自动检测并抓取桌面物体（使用 config.yaml 中配置的后端）
python main.py

# 交互模式：识别后选择要抓取的物体
python main.py --interactive

# 循环抓取（清空桌面）
python main.py --loop

# 只计算坐标，不动机械臂
python main.py --dry-run

# 临时切换检测后端（不修改 config.yaml）
python main.py --detector yolo
python main.py --detector llm
```

## 单模块测试

```bash
python camera.py                           # 预览相机画面，打印内参和 FOV
python llm_detector.py <image_path>        # 测试 LLM 检测
python yolo_detector.py <model.pt> <image> # 测试 YOLO 检测
python robot.py                            # 测试机械臂连接和夹爪
```

## MCP Server（AI 智能体集成）

通过 `mcp_server.py` 将视觉抓取系统暴露为 MCP 工具，支持 OpenClaw、Claude Desktop 等 AI 智能体调用。

```bash
python mcp_server.py
```

环境变量：

| 变量 | 说明 |
|------|------|
| `VISION_PICK_DRY_RUN=1` | 干跑模式，跳过真实机械臂运动 |
| `VISION_PICK_CONFIG` | 自定义配置文件路径（默认 config.yaml） |

提供 3 个工具：

| 工具 | 参数 | 说明 |
|------|------|------|
| `scan_table` | 无 | 扫描桌面，返回物体列表（含评分和抓取建议） |
| `pick_object` | `target_id`（如 `"T1"`） | 抓取指定目标，需先调用 scan_table |
| `get_status` | 无 | 获取系统状态（连接、抓取计数等） |

MCP 客户端配置示例：

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

典型流程：`scan_table` → 用户选择目标 → `pick_object` → 循环。详细用法参见 [CLAW_GUIDE.md](CLAW_GUIDE.md)。

## YOLO 使用指南

YOLO 后端完全在本地运行，无需网络和 API Key，推理速度比云端 LLM 快约 100 倍（10~50ms vs 2~5s）。

### 1. 收集训练图片

先运行系统至少若干次，让 `diagnostics/` 目录积累抓取时的图片，然后：

```bash
python tools/prepare_dataset.py
# 将 diagnostics/ 中的图片分配到 datasets/grape_kiwi/train/ 和 val/
```

### 2. 生成预标签（可选）

```bash
python tools/generate_prelabels.py
# 利用已有的 LLM 检测结果自动生成初始标注，减少手动标注工作量
```

用标注工具（推荐 [LabelImg](https://github.com/HumanSignal/labelImg) 或 [Roboflow](https://roboflow.com)）校验并修正标注。

### 3. 训练模型

```bash
python tools/train_yolo.py
# 训练完成后自动将最优模型复制到 models/grape_kiwi_best.pt
```

训练参数可通过命令行调整，例如：

```bash
python tools/train_yolo.py --epochs 150 --device 0 --model yolov8m.pt
```

### 4. 启用 YOLO 后端

在 `config.yaml` 中修改：

```yaml
detection:
  backend: "yolo"
  yolo:
    model_path: "models/grape_kiwi_best.pt"
    conf_threshold: 0.5
    device: "cpu"   # 或 "0" 使用 GPU
```

或直接通过命令行参数临时切换：

```bash
python main.py --detector yolo
```

---

## 流程

1. 机械臂移动到高位拍照 (z=400mm)，D435 拍摄 RGB+深度 (1920x1080)
2. **检测阶段**（可选其一）：
   - **LLM 后端**：全图发送给云端 Claude（缩放到 640x360），一次调用检测所有物体
   - **YOLO 后端**：本地 YOLOv8 推理（10~50ms），无需网络
3. 按类别过滤（仅保留水果），按物理尺寸和高度过滤杂质
4. 像素坐标转换为机器人坐标 (X, Y)，安全边界检查
5. 多维度评分排序（置信度、可抓取性、边缘距离、深度等），选择最优目标
6. 估算物体高度，动态调整近距检测高度（适应竖放物体）
7. 视觉伺服居中：颜色直方图匹配 + 自适应搜索半径，迭代校正直到 <40px (~2mm)
8. 检测物体朝向，计算最优夹爪 yaw 角（沿短边抓取）
9. 渐进式下降：先在安全高度旋转 yaw，再垂直下降
10. 抓取 → 抬升 → 移动到释放位置 → 释放
11. 放置后验证：回到拍照位重新扫描，对比前后检测结果确认目标物体已消失

## 文件说明

| 文件 | 功能 |
|------|------|
| `config.yaml` | 机器人、相机、检测后端配置参数 |
| `main.py` | 入口 + 配置加载验证 + `--detector` 参数 |
| `pick_system.py` | 系统编排：硬件管理、检测调用、过滤、诊断录制 |
| `pick_workflows.py` | 运行模式（交互/自动/循环）+ 抓取执行 |
| `pick_planner.py` | 抓取规划：高度估计、视觉伺服、坐标计算 |
| `pick_selection.py` | 目标跟踪与多维度评分排序 |
| `pick_models.py` | 数据结构定义 |
| `camera.py` | RealSense D435 封装（color 1920x1080 + depth 1280x720） |
| `llm_detector.py` | 检测后端：OpenRouter Claude 视觉识别（云端） |
| `yolo_detector.py` | 检测后端：本地 YOLOv8 推理（离线，与 LLMDetector 接口兼容） |
| `cv_refine.py` | 深度分割检测、颜色匹配、物体朝向、夹爪 yaw 计算 |
| `coord_transform.py` | 像素→相机系→基座标系坐标变换 + 深度常量定义 |
| `robot.py` | xArm 6 + Gripper G2 控制 |
| `mcp_server.py` | MCP tool 服务端，供 AI 智能体调用（scan/pick/status） |
| `tools/prepare_dataset.py` | 从 diagnostics 提取训练图片到 datasets/ |
| `tools/generate_prelabels.py` | 将 LLM 检测结果转换为 YOLO 标注格式 |
| `tools/train_yolo.py` | YOLOv8 训练脚本，训练后自动导出模型 |
| `datasets/grape_kiwi/data.yaml` | YOLO 数据集配置（2 类） |
| `CLAW_GUIDE.md` | MCP 集成详细指南 |
