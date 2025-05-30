#!/bin/bash
# This script launches multiple AI model servers across different GPUs.
#
# The script performs the following operations:
# 1. Detects which server it's running on (durunda or trumoi)
# 2. Based on the server:
#    - For 'durunda': Launches 16 different experimental models (exp_0_010 through exp_2_110)
#      across GPUs 0-7, with each model in its own screen session
#    - For 'trumoi': Launches 2 models (exp_2_101 and Llama-3.1-Instruct) on GPU 0
# 3. For each model:
#    - Creates a log file
#    - Starts a detached screen session running the model server
#    - Adds model configuration (name, host IP, port, job ID) to the config file
#
# The configuration for all launched models is stored in 'backend/partial_config.jsonl',
# which is used by other components to connect to these model servers.
#
# Usage: ./launcher_partial.sh
#
# Author: Oscar Sainz (oscar.sainz@ehu.eus)

server=$(hostname)
HOST_IP=$(hostname -I | awk '{print $1}')
CONFIG_FILENAME="backend/partial_config.jsonl"

if [ "$server" = "durunda" ] 
then
    # Clean config file
    printf "" >${CONFIG_FILENAME}

    PORT=60010
    GPU=0
    echo "" > .slurm/exp_0_010.log
    screen -dmS exp_0_010 \
        -L -Logfile .slurm/exp_0_010.log \
        ./backend/start_server_partial.sh /proiektuak/ilenia-scratch/models-instruct/exp_0_010-fixed ${PORT} ${GPU}
    echo "Starting exp_0_010 in screen session [exp_0_010] at GPU [${GPU}]. Logs at: .slurm/exp_0_010.log"
    printf '{"model_name": "exp_0_010", "host": "'${HOST_IP}'", "port": '${PORT}', "job_id": "exp_0_010"}\n' >>${CONFIG_FILENAME}

    PORT=60011
    GPU=0
    echo "" > .slurm/exp_0_011.log
    screen -dmS exp_0_011 \
        -L -Logfile .slurm/exp_0_011.log \
        ./backend/start_server_partial.sh /proiektuak/ilenia-scratch/models-instruct/exp_0_011-fixed ${PORT} ${GPU}
    echo "Starting exp_0_011 in screen session [exp_0_011] at GPU [${GPU}]. Logs at: .slurm/exp_0_011.log"
    printf '{"model_name": "exp_0_011", "host": "'${HOST_IP}'", "port": '${PORT}', "job_id": "exp_0_011"}\n' >>${CONFIG_FILENAME}

    PORT=60101
    GPU=1
    echo "" > .slurm/exp_0_101.log
    screen -dmS exp_0_101 \
        -L -Logfile .slurm/exp_0_101.log \
        ./backend/start_server_partial.sh /proiektuak/ilenia-scratch/models-instruct/exp_0_101-fixed ${PORT} ${GPU}
    echo "Starting exp_0_101 in screen session [exp_0_101] at GPU [${GPU}]. Logs at: .slurm/exp_0_101.log"
    printf '{"model_name": "exp_0_101", "host": "'${HOST_IP}'", "port": '${PORT}', "job_id": "exp_0_101"}\n' >>${CONFIG_FILENAME}

    PORT=60110
    GPU=1
    echo "" > .slurm/exp_0_110.log
    screen -dmS exp_0_110 \
        -L -Logfile .slurm/exp_0_110.log \
        ./backend/start_server_partial.sh /proiektuak/ilenia-scratch/models-instruct/exp_0_110-fixed ${PORT} ${GPU}
    echo "Starting exp_0_110 in screen session [exp_0_110] at GPU [${GPU}]. Logs at: .slurm/exp_0_110.log"
    printf '{"model_name": "exp_0_110", "host": "'${HOST_IP}'", "port": '${PORT}', "job_id": "exp_0_110"}\n' >>${CONFIG_FILENAME}

    PORT=60111
    GPU=2
    echo "" > .slurm/exp_0_111.log
    screen -dmS exp_0_111 \
        -L -Logfile .slurm/exp_0_111.log \
        ./backend/start_server_partial.sh /proiektuak/ilenia-scratch/models-instruct/exp_0_111-fixed ${PORT} ${GPU}
    echo "Starting exp_0_111 in screen session [exp_0_111] at GPU [${GPU}]. Logs at: .slurm/exp_0_111.log"
    printf '{"model_name": "exp_0_111", "host": "'${HOST_IP}'", "port": '${PORT}', "job_id": "exp_0_111"}\n' >>${CONFIG_FILENAME}

    PORT=61001
    GPU=2
    echo "" > .slurm/exp_1_001.log
    screen -dmS exp_1_001 \
        -L -Logfile .slurm/exp_1_001.log \
        ./backend/start_server_partial.sh /proiektuak/ilenia-scratch/models-instruct/exp_1_001-fixed ${PORT} ${GPU}
    echo "Starting exp_1_001 in screen session [exp_1_001] at GPU [${GPU}]. Logs at: .slurm/exp_1_001.log"
    printf '{"model_name": "exp_1_001", "host": "'${HOST_IP}'", "port": '${PORT}', "job_id": "exp_1_001"}\n' >>${CONFIG_FILENAME}

    PORT=61010
    GPU=3
    echo "" > .slurm/exp_1_010.log
    screen -dmS exp_1_010 \
        -L -Logfile .slurm/exp_1_010.log \
        ./backend/start_server_partial.sh /proiektuak/ilenia-scratch/models-instruct/exp_1_010-fixed ${PORT} ${GPU}
    echo "Starting exp_1_010 in screen session [exp_1_010] at GPU [${GPU}]. Logs at: .slurm/exp_1_010.log"
    printf '{"model_name": "exp_1_010", "host": "'${HOST_IP}'", "port": '${PORT}', "job_id": "exp_1_010"}\n' >>${CONFIG_FILENAME}

    PORT=61011
    GPU=3
    echo "" > .slurm/exp_1_011.log
    screen -dmS exp_1_011 \
        -L -Logfile .slurm/exp_1_011.log \
        ./backend/start_server_partial.sh /proiektuak/ilenia-scratch/models-instruct/exp_1_011-fixed ${PORT} ${GPU}
    echo "Starting exp_1_011 in screen session [exp_1_011] at GPU [${GPU}]. Logs at: .slurm/exp_1_011.log"
    printf '{"model_name": "exp_1_011", "host": "'${HOST_IP}'", "port": '${PORT}', "job_id": "exp_1_011"}\n' >>${CONFIG_FILENAME}

    PORT=61101
    GPU=4
    echo "" > .slurm/exp_1_101.log
    screen -dmS exp_1_101 \
        -L -Logfile .slurm/exp_1_101.log \
        ./backend/start_server_partial.sh /proiektuak/ilenia-scratch/models-instruct/exp_1_101-fixed ${PORT} ${GPU}
    echo "Starting exp_1_101 in screen session [exp_1_101] at GPU [${GPU}]. Logs at: .slurm/exp_1_101.log"
    printf '{"model_name": "exp_1_101", "host": "'${HOST_IP}'", "port": '${PORT}', "job_id": "exp_1_101"}\n' >>${CONFIG_FILENAME}

    PORT=61110
    GPU=4
    echo "" > .slurm/exp_1_110.log
    screen -dmS exp_1_110 \
        -L -Logfile .slurm/exp_1_110.log \
        ./backend/start_server_partial.sh /proiektuak/ilenia-scratch/models-instruct/exp_1_110-fixed ${PORT} ${GPU}
    echo "Starting exp_1_110 in screen session [exp_1_110] at GPU [${GPU}]. Logs at: .slurm/exp_1_110.log"
    printf '{"model_name": "exp_1_110", "host": "'${HOST_IP}'", "port": '${PORT}', "job_id": "exp_1_110"}\n' >>${CONFIG_FILENAME}

    PORT=61111
    GPU=5
    echo "" > .slurm/exp_1_111.log
    screen -dmS exp_1_111 \
        -L -Logfile .slurm/exp_1_111.log \
        ./backend/start_server_partial.sh /proiektuak/ilenia-scratch/models-instruct/exp_1_111-fixed ${PORT} ${GPU}
    echo "Starting exp_1_111 in screen session [exp_1_111] at GPU [${GPU}]. Logs at: .slurm/exp_1_111.log"
    printf '{"model_name": "exp_1_111", "host": "'${HOST_IP}'", "port": '${PORT}', "job_id": "exp_1_111"}\n' >>${CONFIG_FILENAME}

    PORT=62010
    GPU=5
    echo "" > .slurm/exp_2_010.log
    screen -dmS exp_2_010 \
        -L -Logfile .slurm/exp_2_010.log \
        ./backend/start_server_partial.sh /proiektuak/ilenia-scratch/models-instruct/exp_2_010-fixed ${PORT} ${GPU}
    echo "Starting exp_2_010 in screen session [exp_2_010] at GPU [${GPU}]. Logs at: .slurm/exp_2_010.log"
    printf '{"model_name": "exp_2_010", "host": "'${HOST_IP}'", "port": '${PORT}', "job_id": "exp_2_010"}\n' >>${CONFIG_FILENAME}

    PORT=62100
    GPU=6
    echo "" > .slurm/exp_2_100.log
    screen -dmS exp_2_100 \
        -L -Logfile .slurm/exp_2_100.log \
        ./backend/start_server_partial.sh /proiektuak/ilenia-scratch/models-instruct/exp_2_100-fixed ${PORT} ${GPU}
    echo "Starting exp_2_100 in screen session [exp_2_100] at GPU [${GPU}]. Logs at: .slurm/exp_2_100.log"
    printf '{"model_name": "exp_2_100", "host": "'${HOST_IP}'", "port": '${PORT}', "job_id": "exp_2_100"}\n' >>${CONFIG_FILENAME}

    PORT=62011
    GPU=6
    echo "" > .slurm/exp_2_011.log
    screen -dmS exp_2_011 \
        -L -Logfile .slurm/exp_2_011.log \
        ./backend/start_server_partial.sh /proiektuak/ilenia-scratch/models-instruct/exp_2_011-fixed ${PORT} ${GPU}
    echo "Starting exp_2_011 in screen session [exp_2_011] at GPU [${GPU}]. Logs at: .slurm/exp_2_011.log"
    printf '{"model_name": "exp_2_011", "host": "'${HOST_IP}'", "port": '${PORT}', "job_id": "exp_2_011"}\n' >>${CONFIG_FILENAME}

    PORT=62111
    GPU=7
    echo "" > .slurm/exp_2_111.log
    screen -dmS exp_2_111 \
        -L -Logfile .slurm/exp_2_111.log \
        ./backend/start_server_partial.sh /proiektuak/ilenia-scratch/models-instruct/exp_2_111-fixed ${PORT} ${GPU} # ! Change GPU accordingly! (check .slurm/reserved_trumoi)
    echo "Starting exp_2_111 in screen session [exp_2_111] at GPU [${GPU}]. Logs at: .slurm/exp_2_111.log"
    printf '{"model_name": "exp_2_111", "host": "'${HOST_IP}'", "port": '${PORT}', "job_id": "exp_2_111"}\n' >>${CONFIG_FILENAME}

    PORT=62110
    GPU=7
    echo "" > .slurm/exp_2_110.log
    screen -dmS exp_2_110 \
        -L -Logfile .slurm/exp_2_110.log \
        ./backend/start_server_partial.sh /proiektuak/ilenia-scratch/models-instruct/exp_2_110-fixed ${PORT} ${GPU}
    echo "Starting exp_2_110 in screen session [exp_2_110] at GPU [${GPU}]. Logs at: .slurm/exp_2_110.log"
    printf '{"model_name": "exp_2_110", "host": "'${HOST_IP}'", "port": '${PORT}', "job_id": "exp_2_110"}\n' >>${CONFIG_FILENAME}

elif [ "$server" = "trumoi" ]
then

    echo "" > .slurm/exp_2_101.log
    PORT=8001
    GPU=0
    screen -dmS exp_2_101 \
        -L -Logfile .slurm/exp_2_101.log \
        ./backend/start_server_partial.sh /proiektuak/ilenia-scratch/models-instruct/exp_2_101-fixed ${PORT} ${GPU}
    echo "Starting exp_2_101 in screen session [exp_2_101] at GPU [${GPU}]. Logs at: .slurm/exp_2_101.log"
    printf '{"model_name": "exp_2_101", "host": "'${HOST_IP}'", "port": '${PORT}', "job_id": "exp_2_101"}\n' >>${CONFIG_FILENAME}

    echo "" > .slurm/Llama-3.1-Instruct.log
    PORT=8003
    GPU=0
    screen -dmS Llama-3.1-Instruct \
        -L -Logfile .slurm/Llama-3.1-Instruct.log \
        ./backend/start_server_partial.sh meta-llama/Llama-3.1-8B-Instruct ${PORT} ${GPU} # ! Change GPU accordingly! (check .slurm/reserved_trumoi)
    echo "Starting Llama-3.1-Instruct in screen session [meta-llama/Llama-3.1-8B-Instruct] at GPU [${GPU}]. Logs at: .slurm/Llama-3.1-Instruct.log"
    printf '{"model_name": "Llama-3.1-8B-Instruct", "host": "'${HOST_IP}'", "port": '${PORT}', "job_id": "Llama-3.1-Instruct"}\n' >>${CONFIG_FILENAME}

else
    echo "Server not recognized. Exiting."
    exit 1
fi




