#!/bin/bash
# ZEquant 每日定时任务安装脚本
# 安装后每天 15:30（交易日）自动运行：拉数据 → 出信号 → 发邮件
#
# 用法:
#   bash scripts/install_cron.sh              # 安装
#   bash scripts/install_cron.sh --uninstall  # 卸载
#   bash scripts/install_cron.sh --status     # 查看状态
#   bash scripts/install_cron.sh --run-now    # 立即运行一次

set -e

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PLIST_PATH="$HOME/Library/LaunchAgents/com.zequant.daily.plist"
LOG_DIR="$PROJECT_DIR/data_live/logs"

# 确保日志目录存在
mkdir -p "$LOG_DIR"

# 安装
install() {
    echo "📦 安装 ZEquant 每日定时任务..."

    cat > "$PLIST_PATH" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.zequant.daily</string>

    <key>ProgramArguments</key>
    <array>
        <string>/usr/bin/python3</string>
        <string>-m</string>
        <string>live.runner</string>
        <string>--capital</string>
        <string>50000</string>
        <string>--mode</string>
        <string>full</string>
    </array>

    <key>WorkingDirectory</key>
    <string>$PROJECT_DIR</string>

    <key>StandardOutPath</key>
    <string>$LOG_DIR/daily_stdout.log</string>

    <key>StandardErrorPath</key>
    <string>$LOG_DIR/daily_stderr.log</string>

    <!-- 每个交易日 15:30 运行 -->
    <key>StartCalendarInterval</key>
    <dict>
        <key>Hour</key>
        <integer>15</integer>
        <key>Minute</key>
        <integer>30</integer>
    </dict>

    <!-- 避免任务堆积 -->
    <key>AbandonProcessGroup</key>
    <true/>

    <!-- 如果 Mac 休眠，唤醒后立即执行 -->
    <key>RunAtLoad</key>
    <true/>
</dict>
</plist>
EOF

    chmod 644 "$PLIST_PATH"
    launchctl load "$PLIST_PATH" 2>/dev/null || true

    echo "✅ 已安装: $PLIST_PATH"
    echo "  运行时间: 每天 15:30 (交易日)"
    echo "  日志目录: $LOG_DIR"
    echo ""
    echo "📋 常用命令:"
    echo "  bash scripts/install_cron.sh --status    # 查看状态"
    echo "  bash scripts/install_cron.sh --run-now   # 立即运行"
    echo "  bash scripts/install_cron.sh --uninstall # 卸载"
}

# 卸载
uninstall() {
    echo "🗑️ 卸载定时任务..."
    if [ -f "$PLIST_PATH" ]; then
        launchctl unload "$PLIST_PATH" 2>/dev/null || true
        rm "$PLIST_PATH"
        echo "✅ 已卸载"
    else
        echo "ℹ️ 未安装"
    fi
}

# 查看状态
status() {
    echo "📋 定时任务状态:"
    if launchctl list | grep -q "com.zequant.daily"; then
        echo "  ✅ 已加载"
        launchctl list | grep "com.zequant.daily"
    else
        echo "  ❌ 未加载"
    fi
    echo ""
    echo "📋 最近日志:"
    if [ -f "$LOG_DIR/daily_stdout.log" ]; then
        tail -20 "$LOG_DIR/daily_stdout.log" 2>/dev/null || echo "  无日志"
    else
        echo "  无日志"
    fi
}

# 立即运行
run_now() {
    echo "🚀 立即执行..."
    cd "$PROJECT_DIR"
    python3 -m live.runner --capital 50000 --mode full 2>&1
}

case "${1:-install}" in
    install|--install)
        install
        ;;
    uninstall|--uninstall)
        uninstall
        ;;
    status|--status)
        status
        ;;
    run-now|--run-now)
        run_now
        ;;
    *)
        echo "用法: $0 [install|uninstall|status|run-now]"
        exit 1
        ;;
esac
