#!/bin/bash


# Google Drive download links
EVALUATOR_T2M_URL="https://drive.google.com/file/d/19C_eiEr0kMGlYVJy_yFL6_Dhk3RvmwhM/view?usp=sharing"
EVALUATOR_KIT_URL="https://drive.google.com/file/d/1TKIZ3TSSZawpilC-7Kw7Ws4sNNuzb49p/view?usp=drive_link"
GLOVE_DATA_URL="https://drive.google.com/file/d/1cmXKUT31pqd7_XpJAiWEo1K81TMYHA5n/view?usp=sharing"
PRETRAINED_GDOWN_URL="https://drive.google.com/file/d/1n9s8l3Xo2mLh7e5j6a9Zt8vXqj5kH4/view?usp=sharing"

# Hugging Face models
CLIP_MODEL_NAME="openai/clip-vit-base-patch32"
PRETRAINED_HF_REPO="heng-li/MotionHiFlow"

# Hugging Face endpoints
HF_ENDPOINTS=(
    "https://hf-mirror.com"
    "https://huggingface.co"
)


# Define help function
function help() {
    echo "Usage: bash prepare.sh <COMMAND>"
    echo ""
    echo "Arguments:"
    echo "  COMMAND: all, evaluator, glove, clip, or pretrained"
    echo "  EXTRA: additional parameters appended directly to the python command"

    echo ""
    echo "Example:"
    echo "  bash prepare.sh all"
}

function check_gdown() {
    if ! command -v gdown &> /dev/null; then
        echo "gdown could not be found. Please install it with 'pip install gdown' and try again."
        exit 1
    fi
}

# evaluator preparation
function check_evaluator() {
    if [[ -f "deps/evaluators/t2m/text_mot_match/model/finest.tar" \
      && -f "deps/evaluators/t2m/Comp_v6_KLD005/meta/mean.npy" \
      && -f "deps/evaluators/t2m/Comp_v6_KLD005/meta/std.npy" \
      && -f "deps/evaluators/kit/text_mot_match/model/finest.tar" \
      && -f "deps/evaluators/kit/Comp_v6_KLD005/meta/mean.npy" \
      && -f "deps/evaluators/kit/Comp_v6_KLD005/meta/std.npy" ]]; then
        echo "Evaluators already prepared. Skipping download."
        return 0
    fi
    return 1
}

function prepare_evaluator() {
    check_gdown
    echo "Preparing evaluator..."

    mkdir -p deps/evaluators/t2m
    echo "Downloading T2M evaluator..."
    gdown "$EVALUATOR_T2M_URL" -O deps/evaluators/t2m/humanml3d_evaluator.zip
    echo "Unzipping T2M evaluator..."
    unzip deps/evaluators/t2m/humanml3d_evaluator.zip -d deps/evaluators/t2m
    echo "Cleaning up..."
    rm deps/evaluators/t2m/humanml3d_evaluator.zip

    mkdir -p deps/evaluators/kit
    echo "Downloading KIT evaluator..."
    gdown "$EVALUATOR_KIT_URL" -O deps/evaluators/kit/kit_evaluator.zip
    echo "Unzipping KIT evaluator..."
    unzip deps/evaluators/kit/kit_evaluator.zip -d deps/evaluators/kit
    echo "Cleaning up..."
    rm deps/evaluators/kit/kit_evaluator.zip
}


# glove preparation
function check_glove() {
    if [[ -f "deps/glove/our_vab_data.npy" \
      && -f "deps/glove/our_vab_idx.pkl" \
      && -f "deps/glove/our_vab_words.pkl" ]]; then
        echo "GloVe data already prepared. Skipping download."
        return 0
    fi
    return 1
}

function prepare_glove() {
    check_gdown
    echo "Preparing GloVe data..."

    mkdir -p deps/glove
    echo "Downloading GloVe data..."
    gdown "$GLOVE_DATA_URL" -O deps/glove/glove_data.zip
    echo "Unzipping GloVe data..."
    unzip deps/glove/glove_data.zip -d deps/glove
    echo "Cleaning up..."
    rm deps/glove/glove_data.zip
}


# CLIP preparation
function prepare_clip() {
    echo "Preparing CLIP model..."

    mkdir -p deps

    if ! command -v curl &> /dev/null; then
        echo "curl could not be found. Please install curl and try again."
        return 1
    fi

    check_hf || return 1

    local model_dir="deps/clip-vit-base-patch32"
    local timeout=5
    local chosen_endpoint=""
    local endpoint

    for endpoint in "${HF_ENDPOINTS[@]}"; do
        if curl -L --connect-timeout "$timeout" --max-time "$timeout" --silent --show-error \
            --output /dev/null "$endpoint"; then
            chosen_endpoint="$endpoint"
            break
        fi
        echo "Failed to connect to $endpoint"
    done

    if [[ -z "$chosen_endpoint" ]]; then
        echo "Failed to connect to all endpoints, please check your network settings."
        return 1
    fi

    echo "Successfully connected to $chosen_endpoint, start downloading CLIP model."

    if HF_ENDPOINT="$chosen_endpoint" hf download "$CLIP_MODEL_NAME" \
        --repo-type model \
        --exclude "*.h5" \
        --exclude "*.ot" \
        --exclude "*.msgpack" \
        --exclude "*.safetensors" \
        --local-dir "$model_dir"; then
        echo "CLIP model preparation complete."
        return 0
    fi

    rm -rf "$tmp_dir"
    echo "Failed to download CLIP model."
    return 1
}


# pretrained model preparation
function check_hf() {
    if ! command -v hf &> /dev/null; then
        echo 'hf CLI could not be found. Please install it with `pip install -U huggingface_hub` and try again.'
        return 1
    fi
    return 0
}

function check_pretrained() {
    function check_one() {
        local path="$1"
        if [[ -f "$path/config.yaml" && -f "$path/checkpoints/net_best_fid.tar" ]]; then
            return 0
        fi
        return 1
    }
    check_list=(
        "t2m_vae_agcn_16d"
        "t2m_tmdit_16d"
        "kit_vae_agcn_16d"
        "kit_tmdit_16d"
    )
    for model in "${check_list[@]}"; do
        if ! check_one "logs/$model"; then
            return 1
        fi
    done
    echo "All pretrained models already prepared. Skipping download."
    return 0
}

function sync_pretrained_from_dir() {
    local source_dir="$1"
    local source_root=""
    local checkpoints_dir=""

    if [[ -d "$source_dir/checkpoints" ]]; then
        checkpoints_dir="$source_dir/checkpoints"
    elif [[ -d "$source_dir/logs" ]]; then
        checkpoints_dir="$source_dir/logs"
    else
        checkpoints_dir="$source_dir"
    fi

    if [[ ! -d "$checkpoints_dir" ]]; then
        echo "Could not find extracted pretrained model directory in '$source_dir'."
        return 1
    fi

    mkdir -p logs
    shopt -s nullglob
    for source_root in "$checkpoints_dir"/*; do
        if [[ -d "$source_root" ]]; then
            local target_dir="logs/$(basename "$source_root")"
            echo "Syncing $(basename "$source_root") to $target_dir ..."
            rm -rf "$target_dir"
            cp -a "$source_root" "$target_dir"
        fi
    done
    shopt -u nullglob

    return 0
}

function download_pretrained_with_gdown() {
    local archive_path
    local extract_dir

    archive_path=$(mktemp /tmp/motionhiflow_pretrained_XXXXXX.zip)
    extract_dir=$(mktemp -d /tmp/motionhiflow_pretrained_XXXXXX)

    echo "Downloading pretrained models with gdown..."
    if ! gdown "$PRETRAINED_GDOWN_URL" -O "$archive_path"; then
        echo "gdown download failed."
        rm -f "$archive_path"
        rm -rf "$extract_dir"
        return 1
    fi

    echo "Unzipping pretrained models..."
    if ! unzip -o "$archive_path" -d "$extract_dir"; then
        echo "Failed to unzip pretrained models archive."
        rm -f "$archive_path"
        rm -rf "$extract_dir"
        return 1
    fi

    if ! sync_pretrained_from_dir "$extract_dir"; then
        rm -f "$archive_path"
        rm -rf "$extract_dir"
        return 1
    fi

    rm -f "$archive_path"
    rm -rf "$extract_dir"
    return 0
}

function download_pretrained_with_hf() {
    local tmp_dir
    local endpoint

    check_hf || return 1

    tmp_dir=$(mktemp -d /tmp/motionhiflow_hf_XXXXXX)

    for endpoint in "${HF_ENDPOINTS[@]}"; do
        echo "Trying Hugging Face endpoint: $endpoint"
        if HF_ENDPOINT="$endpoint" hf download "$PRETRAINED_HF_REPO" \
            --repo-type model \
            --include "checkpoints/*" \
            --local-dir "$tmp_dir" \
            --force-download; then
            if sync_pretrained_from_dir "$tmp_dir"; then
                rm -rf "$tmp_dir"
                return 0
            fi
        fi

        echo "******* Download from $endpoint failed, trying next endpoint... *******"
        rm -rf "$tmp_dir"
        tmp_dir=$(mktemp -d /tmp/motionhiflow_hf_XXXXXX)
    done

    rm -rf "$tmp_dir"
    return 1
}

function prepare_pretrained() {
    echo "Preparing pretrained models..."

    if command -v gdown &> /dev/null; then
        if download_pretrained_with_gdown; then
            echo "Pretrained models downloaded successfully with gdown."
            echo "Pretrained model preparation complete."
            return 0
        fi
        echo "gdown download failed. Falling back to Hugging Face..."
    else
        echo "gdown not found. Falling back to Hugging Face..."
    fi

    if download_pretrained_with_hf; then
        echo "Pretrained models downloaded successfully from Hugging Face."
        echo "Pretrained model preparation complete."
        return 0
    fi

    echo "Failed to prepare pretrained models with both gdown and Hugging Face."
    echo "You can manually download all folders from:"
    echo "https://huggingface.co/heng-li/MotionHiFlow/tree/main/checkpoints"
    echo "Pretrained model preparation complete."
    return 1
}

case "$1" in
    all)
        check_evaluator || prepare_evaluator
        check_glove || prepare_glove
        prepare_clip
        check_pretrained || prepare_pretrained
        ;;
    evaluator)
        prepare_evaluator
        ;;
    glove)
        prepare_glove
        ;;
    clip)
        prepare_clip
        ;;
    pretrained)
        prepare_pretrained
        ;;
    *)
        help
        ;;
esac