BEST_GPU=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | awk '{print $1 " " NR-1}' | sort -nr | head -n1 | awk '{print $2}')

# Display the selected GPU
echo "Selected GPU: $BEST_GPU"

# Set the selected GPU to CUDA_VISIBLE_DEVICES
export CUDA_VISIBLE_DEVICES=$BEST_GPU

# Run your Python script
python train.py
