#!/bin/bash

# WrightEagle (Soccer Simulation League 2D)
# BASE SOURCE CODE RELEASE 2016
# Copyright (c) 1998-2016 WrightEagle 2D Soccer Simulation Team,
#                         Multi-Agent Systems Lab.,
#                         School of Computer Science and Technology,
#                         University of Science and Technology of China
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#     * Neither the name of the WrightEagle 2D Soccer Simulation Team nor the
#       names of its contributors may be used to endorse or promote products
#       derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL WrightEagle 2D Soccer Simulation Team BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

HOST="localhost"
PORT="6000"
VERSION="Debug"
BINARY="WEBase"
TEAM_NAME="WEBase"
PLAYER_COUNT=11
ENABLE_NN=0
MATCH_DURATION=0
RUN_OPPONENT=0
WAIT_FOR_EXTERNAL_OPPONENT=0
OPPONENT_TEAM="Opponent"

# 解析参数
while getopts "h:p:v:b:t:n:m:ure" flag; do
	case "$flag" in
	h) HOST=$OPTARG ;;
	p) PORT=$OPTARG ;;
	v) VERSION=$OPTARG ;;
	b) BINARY=$OPTARG ;;
	t) TEAM_NAME=$OPTARG ;;
	n) PLAYER_COUNT=$OPTARG ;;
	m) MATCH_DURATION=$OPTARG ;;
	u) ENABLE_NN=1 ;;
	r) RUN_OPPONENT=1 ;;
	e) WAIT_FOR_EXTERNAL_OPPONENT=1 ;;
	esac
done

echo "=========================================="
echo "WrightEagle Start Script"
echo "=========================================="
echo "Team: $TEAM_NAME"
echo "Host: $HOST:$PORT"
echo "Version: $VERSION"
echo "NN Mode: $ENABLE_NN"
echo "Run Opponent: $RUN_OPPONENT"
echo "Wait for External Opp: $WAIT_FOR_EXTERNAL_OPPONENT"
if [ $MATCH_DURATION -gt 0 ]; then
	echo "Match Duration: ${MATCH_DURATION} minutes"
fi
echo "=========================================="

# 清理旧进程
echo "[1/6] Cleaning up old processes..."
pkill -9 -f "rcssserver|rcssmonitor" 2>/dev/null
sleep 1

# 设置 NN 环境变量
if [ $ENABLE_NN -eq 1 ]; then
	export USE_NN=1
	if [ -z "$NN_MODEL" ]; then
		if [ -f "models/value_nn.bin" ]; then
			export NN_MODEL="models/value_nn.bin"
			echo "[NN] Using model: $NN_MODEL"
		fi
	fi
fi

# 启动服务器
echo "[2/6] Starting rcssserver..."
rcssserver server::auto_mode=on server::kick_off_wait=30 &
sleep 3

# 启动 WEBase (左侧)
echo "[3/6] Starting $TEAM_NAME (left side, $PLAYER_COUNT players + 1 goalie)..."
CLIENT="./$VERSION/$BINARY"
LOG_DIR="Logfiles"
mkdir $LOG_DIR 2>/dev/null
SLEEP_TIME=0.1

COACH_PORT=$(expr $PORT + 1)
OLCOACH_PORT=$(expr $PORT + 2)
N_PARAM="-team_name $TEAM_NAME -host $HOST -port $PORT -coach_port $COACH_PORT -olcoach_port $OLCOACH_PORT -log_dir $LOG_DIR"
G_PARAM="$N_PARAM -goalie on"

echo ">>>>>>>>>>>>>>>>>>>>>> $TEAM_NAME Goalie: 1"
$CLIENT $G_PARAM &
sleep 5

i=2
while [ $i -le $PLAYER_COUNT ]; do
	echo ">>>>>>>>>>>>>>>>>>>>>> $TEAM_NAME Player: $i"
	$CLIENT $N_PARAM &
	sleep $SLEEP_TIME
	i=$(expr $i + 1)
done
sleep 3

echo ">>>>>>>>>>>>>>>>>>>>>> $TEAM_NAME Coach"
$CLIENT $N_PARAM -coach &
sleep 1

# 等待外部对手启动
if [ $WAIT_FOR_EXTERNAL_OPPONENT -eq 1 ]; then
	echo "[4/6] Waiting for external opponent to connect..."
	echo "       Please start opponent team manually, e.g.:"
	echo "       cd /Users/chenmao/Code/wrighteagle2 && ./start.sh -t TeamB -h localhost"
	while true; do
		sleep 5
		# 检查是否有对手连接
		if pgrep -f "rcssserver" > /dev/null; then
			echo "       Server still running, waiting for opponent..."
		else
			echo "       Server stopped, exiting..."
			exit 1
		fi
		# 如果设置了定时，等待时间到了就退出
		if [ $MATCH_DURATION -gt 0 ]; then
			break
		fi
	done
fi

# 启动对手 (右侧)
if [ $RUN_OPPONENT -eq 1 ]; then
	echo "[4/6] Starting $OPPONENT_TEAM (right side, $PLAYER_COUNT players + 1 goalie)..."

	# 对手使用不同的教练端口
	OPP_COACH_PORT=$(expr $PORT + 3)
	OPP_OLCOACH_PORT=$(expr $PORT + 4)
	OPP_N_PARAM="-team_name $OPPONENT_TEAM -host $HOST -port $PORT -coach_port $OPP_COACH_PORT -olcoach_port $OPP_OLCOACH_PORT -log_dir $LOG_DIR"
	OPP_G_PARAM="$OPP_N_PARAM -goalie on"

	echo ">>>>>>>>>>>>>>>>>>>>>> $OPPONENT_TEAM Goalie: 1"
	$CLIENT $OPP_G_PARAM &
	sleep 5

	i=2
	while [ $i -le $PLAYER_COUNT ]; do
		echo ">>>>>>>>>>>>>>>>>>>>>> $OPPONENT_TEAM Player: $i"
		$CLIENT $OPP_N_PARAM &
		sleep $SLEEP_TIME
		i=$(expr $i + 1)
	done
	sleep 3

	echo ">>>>>>>>>>>>>>>>>>>>>> $OPPONENT_TEAM Coach"
	$CLIENT $OPP_N_PARAM -coach &
	sleep 1
fi

# 启动 Monitor
echo "[5/6] Starting rcssmonitor..."
DYLD_LIBRARY_PATH=/usr/local/lib rcssmonitor &
sleep 1

echo "[6/6] Match started!"
echo "=========================================="

# 定时停止
if [ $MATCH_DURATION -gt 0 ]; then
	echo "Match will end in ${MATCH_DURATION} minutes..."
	sleep $(expr $MATCH_DURATION \* 60)
	echo "Time's up! Stopping match..."

	# 先用 SIGTERM 让进程优雅退出（保存数据）
	pkill -f "rcssserver|WEBase|rcssmonitor" 2>/dev/null
	sleep 5

	# 强制杀死残留进程
	pkill -9 -f "rcssserver|WEBase|rcssmonitor" 2>/dev/null
	echo "All processes killed."
else
	echo "Match running... Press Ctrl+C to stop."
	wait
fi
