python train_registration_lumir.py --model EOIR --batch_size 1 --dataset lumir --gpu_id 0 \
                                   --epochs 201 --reg_w 5.0 --ncc_w 1.0 --lr 4e-4 \
                                   --start_channel 16 \
                                   --json_path ./LUMIR_L2R24_TrainVal/LUMIR_dataset.json \
                                   --datasets_path ./LUMIR_L2R24_TrainVal
