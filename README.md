# Vision Pick

基于 xArm 6 + RealSense D435 + 云端 LLM 视觉的桌面物体抓取系统。

## 硬件

- xArm 6 机械臂 (192.168.1.60)
- Gripper G2 夹爪 (最大开口 84mm)
- Intel RealSense D435 深度相机 (eye-in-hand)

## 安装

```bash
conda activate py312
pip install -r requirements.txt
```

在 `.env` 中配置 API Key：
```
OPENROUTER_KEY=your_key_here
```

## 使用

```bash
# 自动检测并抓取桌面物体
python main.py

# 交互模式：LLM 识别后选择要抓取的物体
python main.py --interactive

# 循环抓取（清空桌面）
python main.py --loop

# 只计算坐标，不动机械臂
python main.py --dry-run
```

## 单模块测试

```bash
python camera.py          # 预览相机画面，打印内参和 FOV
python llm_detector.py <image_path>  # 测试 LLM 检测
python robot.py           # 测试机械臂连接和夹爪
```

## 流程

1. 机械臂移动到高位拍照 (z=400mm)，D435 拍摄 RGB+深度 (1920x1080)
2. 全图发送给云端 Claude（缩放到 640x360），一次 LLM 调用检测所有物体
3. 按类别过滤（仅保留水果），按物理尺寸和高度过滤杂质
4. 像素坐标转换为机器人坐标 (X, Y)，安全边界检查
5. 多维度评分排序（置信度、可抓取性、边缘距离、深度等），选择最优目标
6. 估算物体高度，动态调整近距检测高度（适应竖放物体）
7. 视觉伺服居中：颜色直方图匹配 + 自适应搜索半径，迭代校正直到 <40px (~2mm)
8. 检测物体物理尺寸，超出夹爪行程则跳过
9. 检测物体朝向，计算最优夹爪 yaw 角（沿短边抓取）
10. 渐进式下降：先在安全高度旋转 yaw，再垂直下降
11. 抓取 → 抬升 → 移动到释放位置 → 释放 → 复位

## 文件说明

| 文件 | 功能 |
|------|------|
| config.yaml | 机器人、相机、LLM 配置参数 |
| main.py | 入口 + 配置加载验证 |
| pick_system.py | 系统编排：硬件管理、LLM 扫描、过滤、诊断录制 |
| pick_workflows.py | 运行模式（交互/自动/循环）+ 抓取执行 |
| pick_planner.py | 抓取规划：高度估计、视觉伺服、坐标计算 |
| pick_selection.py | 目标跟踪与多维度评分排序 |
| pick_models.py | 数据结构定义 |
| camera.py | RealSense D435 封装（color 1920x1080 + depth 1280x720） |
| llm_detector.py | OpenRouter Claude 视觉识别（全图多目标检测） |
| cv_refine.py | 深度分割检测、颜色匹配、物体朝向、夹爪 yaw 计算 |
| coord_transform.py | 像素→相机系→基座标系坐标变换 + 深度常量定义 |
| robot.py | xArm 6 + Gripper G2 控制 |
