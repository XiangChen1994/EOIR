#!/usr/bin/env bash
#bash ./build.sh

# docker load --input eoir.tar.gz

docker run --rm  \
        --ipc=host \
        --memory 16g \
        --mount type=bind,source=./LUMIR_L2R24_TrainVal/LUMIR_dataset.json,target=/LUMIR_dataset.json \
        --mount type=bind,source=./LUMIR_L2R24_TrainVal/,target=/input \
        --mount type=bind,source=./docker/DockerImage_EOIR/output/,target=/output \
        eoir

# #!/usr/bin/env bash
# #bash ./build.sh
# docker load --input EOIR.tar.gz

# docker run --rm  \
#         --ipc=host \
#         --memory 256g \
#         --mount type=bind,source=[PATH for .json dataset file],target=/LUMIR_dataset.json \
#         --mount type=bind,source=[Directory of input images],target=/input \
#         --mount type=bind,source=[Directory of output predictions],target=/output \
#         eoir