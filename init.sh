#!/bin/bash

SCRIPT_DIR="$( cd "$(dirname "$0")" >/dev/null 2>&1 || exit ; pwd -P )"

echo "script directory is: ${SCRIPT_DIR}"
echo "run 'help' to list available functions"

function log_green() { printf "\x1B[32m>> %s\x1B[39m\n" "$1"; }
function log_red() { printf "\x1B[31m>> %s\x1B[39m\n" "$1"; }
function log_blue() { printf "\x1B[94m>> %s\x1B[39m\n" "$1"; }
function log_green_text() { printf "\x1B[32m%s\x1B[39m" "$1"; }
function log_red_text() { printf "\x1B[31m%s\x1B[39m" "$1"; }
function log_blue_text() { printf "\x1B[94m%s\x1B[39m" "$1"; }

function help() {
    log_green "Matrix permanant shell functions:"

    log_red "Commands to prepare environment:"
    log_green "$(log_blue_text "load_modules"): Loads necessary cuda and gcc modules."
    log_green "$(log_blue_text "compile"): Compile the code with nvcc." 
    log_green "$(log_blue_text "set_threads"): PARAMETERS ~ {THREAD_NUM}"
    log_green "$(log_blue_text "set_matrix"): PARAMETERS ~ {PATH_TO_MATRIX}"

    log_red "Commands to run exact algorithms:"
    log_green "$(log_blue_text "gpu_perman64_xlocal"): PARAMETERS ~ {GRID_DIM}, {BLOCK_DIM}"
    log_green "$(log_blue_text "gpu_perman64_xlocal_sparse"): PARAMETERS ~ {GRID_DIM}, {BLOCK_DIM}"
    log_green "$(log_blue_text "gpu_perman64_xshared"): PARAMETERS ~ {GRID_DIM}, {BLOCK_DIM}"
    log_green "$(log_blue_text "gpu_perman64_xshared_sparse"): PARAMETERS ~ {GRID_DIM}, {BLOCK_DIM}"
    log_green "$(log_blue_text "gpu_perman64_xshared_coalescing"): PARAMETERS ~ {GRID_DIM}, {BLOCK_DIM}"
    log_green "$(log_blue_text "gpu_perman64_xshared_coalescing_sparse"): PARAMETERS ~ {GRID_DIM}, {BLOCK_DIM}"
    log_green "$(log_blue_text "gpu_perman64_xshared_coalescing_mshared"): PARAMETERS ~ {GRID_DIM}, {BLOCK_DIM}"
    log_green "$(log_blue_text "gpu_perman64_xshared_coalescing_mshared_sparse"): PARAMETERS ~ {GRID_DIM}, {BLOCK_DIM}"
    log_green "$(log_blue_text "parallel_perman64"): "
    log_green "$(log_blue_text "parallel_perman64_sparse"): "
    log_green "$(log_blue_text "gpu_perman64_xglobal"): PARAMETERS ~ {GRID_DIM}, {BLOCK_DIM}"

    log_red "Commands to run approximation algorithms:"
    log_green "$(log_blue_text "gpu_perman64_rasmussen"): PARAMETERS ~ {NUMBER_OF_TIMES}"
    log_green "$(log_blue_text "gpu_perman64_rasmussen_sparse"): PARAMETERS ~ {NUMBER_OF_TIMES}"
    log_green "$(log_blue_text "gpu_perman64_approximation"): PARAMETERS ~ {NUMBER_OF_TIMES}, {SCALE_INTERVALS}, {SCALE_TIMES}"
    log_green "$(log_blue_text "gpu_perman64_approximation_sparse"): PARAMETERS ~ {NUMBER_OF_TIMES}, {SCALE_INTERVALS}, {SCALE_TIMES}"
    log_green "$(log_blue_text "rasmussen"): PARAMETERS ~ {NUMBER_OF_TIMES}"
    log_green "$(log_blue_text "rasmussen_sparse"): PARAMETERS ~ {NUMBER_OF_TIMES}"
    log_green "$(log_blue_text "approximation_perman64"): PARAMETERS ~ {NUMBER_OF_TIMES}, {SCALE_INTERVALS}, {SCALE_TIMES}"
    log_green "$(log_blue_text "approximation_perman64_sparse"): PARAMETERS ~ {NUMBER_OF_TIMES}, {SCALE_INTERVALS}, {SCALE_TIMES}"
}

function load_modules() {
    module load gcc/5.3.0
    module load cuda/10.0
}

function compile() {
    nvcc main.cu -O3 -Xcompiler -fopenmp
}

function compile_local() {
    g++ -fopenmp main.cpp
}

function set_threads() {
    THREADS=$1
    export OMP_NUM_THREADS=${THREADS}
}

function set_matrix() {
    MATRIX=$1
    cp ${MATRIX} ${SCRIPT_DIR}/matrix.txt
}



function gpu_perman64_xlocal() {
    GRID_DIM=$1
    BLOCK_DIM=$2
    ./a.out 1 ${GRID_DIM} ${BLOCK_DIM}
}

function gpu_perman64_xlocal_sparse() {
    GRID_DIM=$1
    BLOCK_DIM=$2
    ./a.out 2 ${GRID_DIM} ${BLOCK_DIM}
}

function gpu_perman64_xshared() {
    GRID_DIM=$1
    BLOCK_DIM=$2
    ./a.out 3 ${GRID_DIM} ${BLOCK_DIM}
}

function gpu_perman64_xshared_sparse() {
    GRID_DIM=$1
    BLOCK_DIM=$2
    ./a.out 4 ${GRID_DIM} ${BLOCK_DIM}
}

function gpu_perman64_xshared_coalescing() {
    GRID_DIM=$1
    BLOCK_DIM=$2
    ./a.out 5 ${GRID_DIM} ${BLOCK_DIM}
}

function gpu_perman64_xshared_coalescing_sparse() {
    GRID_DIM=$1
    BLOCK_DIM=$2
    ./a.out 6 ${GRID_DIM} ${BLOCK_DIM}
}

function gpu_perman64_xshared_coalescing_mshared() {
    GRID_DIM=$1
    BLOCK_DIM=$2
    ./a.out 7 ${GRID_DIM} ${BLOCK_DIM}
}

function gpu_perman64_xshared_coalescing_mshared_sparse() {
    GRID_DIM=$1
    BLOCK_DIM=$2
    ./a.out 8 ${GRID_DIM} ${BLOCK_DIM}
}

function parallel_perman64() {
    ./a.out 9
}

function parallel_perman64_sparse() {
    ./a.out 10
}




function gpu_perman64_rasmussen() {
    NUMBER_OF_TIMES=$1
    ./a.out 11 ${NUMBER_OF_TIMES}
}

function gpu_perman64_rasmussen_sparse() {
    NUMBER_OF_TIMES=$1
    ./a.out 12 ${NUMBER_OF_TIMES}
}

function gpu_perman64_approximation() {
    NUMBER_OF_TIMES=$1
    SCALE_INTERVALS=$2
    SCALE_TIMES=$3
    ./a.out 13 ${NUMBER_OF_TIMES} ${SCALE_INTERVALS} ${SCALE_TIMES}
}

function gpu_perman64_approximation_sparse() {
    NUMBER_OF_TIMES=$1
    SCALE_INTERVALS=$2
    SCALE_TIMES=$3
    ./a.out 14 ${NUMBER_OF_TIMES} ${SCALE_INTERVALS} ${SCALE_TIMES}
}

function rasmussen() {
    NUMBER_OF_TIMES=$1
    ./a.out 15 ${NUMBER_OF_TIMES}
}

function rasmussen_sparse() {
    NUMBER_OF_TIMES=$1
    ./a.out 16 ${NUMBER_OF_TIMES}
}

function approximation_perman64() {
    NUMBER_OF_TIMES=$1
    SCALE_INTERVALS=$2
    SCALE_TIMES=$3
    ./a.out 17 ${NUMBER_OF_TIMES} ${SCALE_INTERVALS} ${SCALE_TIMES}
}

function approximation_perman64_sparse() {
    NUMBER_OF_TIMES=$1
    SCALE_INTERVALS=$2
    SCALE_TIMES=$3
    ./a.out 18 ${NUMBER_OF_TIMES} ${SCALE_INTERVALS} ${SCALE_TIMES}
}


function gpu_perman64_xglobal() {
    GRID_DIM=$1
    BLOCK_DIM=$2
    ./a.out 19 ${GRID_DIM} ${BLOCK_DIM}
}
