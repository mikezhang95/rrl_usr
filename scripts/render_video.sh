
# set up mujoco_py
export MUJOCO_GL=egl
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco210/bin

# set up cuda
export CUDA_DEVICE_ORDER=PCI_BUS_ID

# # cartpole_blance/cartpole_swingup
# base_dir="outputs/cartpole_swingup-adv"
# perturb_param_list="pole_length"
# perturb_min_list="0.3"
# perturb_max_list="3.0"
 
# # walker_stand/walker_walk
# base_dir="outputs/walker_stand-adv"
# perturb_param_list="thigh_length"
# perturb_min_list="0.1"
# perturb_max_list="0.7"

# quadruped_walk/quadruped_run
base_dir="outputs/quadruped_walk-adv"
perturb_param_list="contact_friction"
perturb_min_list="0.1"
perturb_max_list="4.5"

perturb_param_list=($perturb_param_list)
perturb_min_list=($perturb_min_list)
perturb_max_list=($perturb_max_list)
length=${#perturb_param_list[@]} 


cuda_id=0
for seed in 12345; do
    export CUDA_VISIBLE_DEVICES=${cuda_id}
    cuda_id=$(($cuda_id+1))
    exp_dir=$base_dir/$seed 
    for ((i=0; i<${length}; i++));do
         python video.py \
             --experiments_dir ${exp_dir} \
             --agent_dir ${exp_dir} \
             --num_steps 100 \
             --perturb_param ${perturb_param_list[$i]} \
             --perturb_min ${perturb_min_list[$i]} \
             --perturb_max ${perturb_max_list[$i]} \
             --save_video &
    done
done



