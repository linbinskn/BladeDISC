# Copyright 2022 The BladeDISC Authors. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# !/bin/bash
if [ -f $HOME/.cache/proxy_config ]; then
  source $HOME/.cache/proxy_config
fi
script_dir=$(cd $(dirname "$0"); pwd)
benchmark_repo_dir=$HOME/.cache/torchbenchmark
# cache benchmark repo
if [ ! -d $benchmark_repo_dir ]; then
    git clone -q https://github.com/pai-disc/torchbenchmark.git --recursive $benchmark_repo_dir
fi

# parse the arguments
HARDWARE=$1; shift
JOB=$1; shift
FIELDS=("$@")

# setup for torchbenchmark
pushd $benchmark_repo_dir
# cache venv in benchmark dir
if [ $HARDWARE == "aarch64" ]; then
    cp -r /opt/venv_disc ./venv
fi
python3 -m virtualenv venv --system-site-packages && source venv/bin/activate

# install dependencies
python3 -m pip install -q -r $script_dir/requirements_$HARDWARE.txt
git pull && git checkout main  && git submodule update --init --recursive --depth 1 && python3 install.py --continue_on_fail

pushd $script_dir # pytorch_blade/benchmark/TorchBench
ln -s $benchmark_repo_dir torchbenchmark

# benchmark
# setup benchmark env
export DISC_EXPERIMENTAL_SPECULATION_TLP_ENHANCE=true \
    DISC_CPU_LARGE_CONCAT_NUM_OPERANDS=4 DISC_CPU_ENABLE_EAGER_TRANSPOSE_FUSION=1 \
    TORCHBENCH_ATOL=1e-2 TORCHBENCH_RTOL=1e-2

bench_target=$JOB

if [[ $HARDWARE == "cpu" ]] || [[ $HARDWARE == "aarch64" ]]
then
    binding_cores=0
    config_file=blade_cpu_$JOB.yaml
    results=()
    declare -A threads2cores
    threads2cores=([1]="0" [2]="0-1" [4]="0-3" [8]="0-7")
    for threads in $(echo ${!threads2cores[*]})
    do
        cores=${threads2cores[$threads]}
        result=eval-$HARDWARE-fp32_$threads
        results[${#results[*]}]=$result
        export OMP_NUM_THREADS=$threads GOMP_CPU_AFFINITY=$cores
        taskset -c $cores python3 torchbenchmark/.github/scripts/run-config.py \
                -c $config_file -b ./torchbenchmark/ --output-dir .
	mv eval-cpu-fp32 $result
    done
else
    config_file=blade_cuda_$JOB.yaml
    results=(eval-cuda-fp32 eval-cuda-fp16)
    python3 torchbenchmark/.github/scripts/run-config.py \
            -c $config_file -b ./torchbenchmark/ --output-dir .
fi

# results
date_str=$(date '+%Y%m%d-%H')
oss_link=https://bladedisc-ci.oss-cn-hongkong.aliyuncs.com
oss_dir=oss://bladedisc-ci/TorchBench/${bench_target}/${date_str}
OSSUTIL=ossutil
GH=gh
if [ $HARDWARE == "aarch64" ]; then
    OSSUTIL=ossutil-arm64
    GH=gh_arm64
fi

for result in ${results[@]}
do
    cat ${result}/summary.csv
    curl ${oss_link}/TorchBench/baseline/${result}_${bench_target}.csv -o $result.csv
    /disc/scripts/ci/$OSSUTIL cp -r ${script_dir}/${result} ${oss_dir}/${result}
done

# Default compare fields
if [ ${#FIELDS[@]} -eq 0 ]; then
    if [ $HARDWARE == "cuda" ]; then
        FIELDS=("disc (latency)" "blade (latency)" "dynamo-blade (latency)" "dynamo-disc (latency)")
    else
        FIELDS=("disc (latency)" "dynamo-disc (latency)")
    fi
fi

# performance anaysis
python3 results_anaysis.py -t ${results} -i ${oss_dir} -p ${RELATED_DIFF_PERCENT} -f "${FIELDS[@]}"

if [ -f "ISSUE.md" ]; then
    wget ${oss_link}/download/github/$GH -O gh && chmod +x ./gh && \
    ./gh issue create -F ISSUE.md \
    -t "[TorchBench] Performance Signal Detected" \
    -l Benchmark
fi

popd # $benchmark_repo_dir
popd # BladeDISC/
