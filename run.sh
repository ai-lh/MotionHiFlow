#!/bin/bash


# Define help function
function help() {
    echo "Usage: bash run.sh <COMMAND> <ID> [<EXTRA>]"
    echo ""
    echo "Arguments:"
    echo "  COMMAND: train, eval, or gen"
    echo "  ID: predefined configuration ID"
    echo "  EXTRA: additional parameters appended directly to the python command"

    echo ""
    echo "Available config IDs:"
    if [[ ${#CONFIG_LIST[@]} -eq 0 ]]; then
        echo "  (no predefined configs available)"
    else
        for key in $(printf '%s\n' "${!CONFIG_LIST[@]}" | sort); do
            echo "  ${key}: ${CONFIG_LIST[$key]}"
        done
    fi

    echo ""
    echo "Example:"
    echo "  bash run.sh eval mvae gpu_id=0"
}

# Predefined configuration list; add more key-value pairs as needed
declare -A CONFIG_LIST=(
    [mvae]="id=agcn_16d model=vae"
    [tmdit]="id=16d model=tmdit vae_model.name=t2m_vae_agcn_16d cond_scale=4.5 time_steps=12"
    [mvae-kit]="id=agcn_16d model=vae data=kit"
    [tmdit-kit]="id=16d model=tmdit data=kit vae_model.name=kit_vae_agcn_16d cond_scale=4.0 time_steps=20"
)

# Function: resolve a key to the corresponding config string
function resolve_config() {
    local key="$1"
    if [[ -z "$key" ]]; then
        echo ""
        return
    fi
    if [[ -n "${CONFIG_LIST[$key]}" ]]; then
        echo "${CONFIG_LIST[$key]}"
    else
        # treat the given value as a path or raw argument string
        echo "$key"
    fi
}

case "$1" in
    train)
        # $2 是 ID
        # $3 及之后的参数作为额外参数传给 python
        ID="$2"
        shift 2
        EXTRA_PARAMS=("$@")
        ID=$(resolve_config "$ID")
        python train.py $ID "${EXTRA_PARAMS[@]}"
        ;;
    eval)
        ID="$2"
        shift 2
        EXTRA_PARAMS=("$@")
        ID=$(resolve_config "$ID")
        python eval.py $ID "${EXTRA_PARAMS[@]}"
        ;;
    gen)
        ID="$2"
        shift 2
        EXTRA_PARAMS=("$@")
        ID=$(resolve_config "$ID")
        python gen_t2m.py $ID "${EXTRA_PARAMS[@]}"
        ;;
    *)
        help
        ;;
esac