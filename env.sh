# 在 AutoDL 服务器上 source 此文件，或在 ~/.bashrc 中 source 它
# echo "source /root/autodl-tmp/env.sh" >> ~/.bashrc

export KST_DATA_ROOT=/root/autodl-tmp/dataset
export PYTHONPATH=/root/autodl-tmp/code:$PYTHONPATH

echo "[kst] KST_DATA_ROOT=$KST_DATA_ROOT"
echo "[kst] PYTHONPATH=$PYTHONPATH"
