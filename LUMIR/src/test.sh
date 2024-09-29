python test_registration_lumir.py --model EOIR --batch_size 1 --dataset lumir \
                                  --gpu_id 0 --num_workers 4 --load_ckpt best \
                                  --datasets_path ./LUMIR_L2R24_TrainVal \
                                  --json_path ./LUMIR_L2R24_TrainVal/LUMIR_dataset.json\
                                  --start_channel 16

