#!/bin/bash
echo "==========================================="
echo " 基础设施外观缺陷智能检测系统 v1.0"
echo "==========================================="

# 激活 conda 环境
source activate guanggaoji 2>/dev/null || conda activate guanggaoji 2>/dev/null || echo "[警告] conda 环境激活失败"
echo "[OK] 环境就绪"

# 切换目录
cd "$(dirname "$0")"

# 检查模型
if [ ! -f "pt/yolov8n-seg-cracks-joints.pt" ]; then
    echo "[错误] 模型文件不存在"
    exit 1
fi
echo "[OK] 模型文件存在"

# 检查依赖
python3 -c "import ultralytics, fastapi, cv2" 2>/dev/null || pip install -r requirements.txt
echo "[OK] 依赖就绪"

# 自动打开浏览器（macOS/Linux）
sleep 2 && (open http://localhost:8000 2>/dev/null || xdg-open http://localhost:8000 2>/dev/null) &

# 启动服务
echo "[启动] http://localhost:8000"
python3 -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
