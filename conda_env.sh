#!/bin/sh
env_name=$1
cuda_path=$2

prefix="/home/ma-user/work/users/yantong"
_CONDA_ROOT=$prefix"/miniconda3"

\. "$_CONDA_ROOT/etc/profile.d/conda.sh" || return $?

export anaconda_home=$prefix/miniconda3/envs/$env_name

if [ -z "$cuda_path" ]; then
    export CUDA_HOME="/usr/local/cuda"
else
    export CUDA_HOME="$cuda_path"
fi

export PATH=$CUDA_HOME/bin:$anaconda_home/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$anaconda_home/lib:$LD_LIBRARY_PATH

conda activate "$env_name"


export HF_HOME="/home/ma-user/work/dataset/huggingface"
export HF_DATASETS_CACHE="/home/ma-user/work/dataset/huggingface/datasets"

echo "====================================================="
echo "Conda: $env_name"
echo "CUDA: $CUDA_HOME"
echo "====================================================="
