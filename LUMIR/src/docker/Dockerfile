FROM python:3.8-slim

# Update the pip
RUN python -m pip install --user -U pip

# Create folder : /input /output /json_dir
RUN mkdir -p /input /output /json_dir

# Upgrade pip
RUN pip install --upgrade pip

# Copy necessary files into the Docker image
COPY ./requirements.txt .
COPY ./EOIR.py .
COPY ./infer_EOIR.py .

# Copy pretrained weights into the Docker image
COPY ./pretrained_weights pretrained_weights/

# Install the torch
# RUN pip install torch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip install torch==1.13.0+cpu torchvision==0.14.0+cpu torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cpu -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN python3 -m pip install --user -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

ENTRYPOINT ["python3","-u","./infer_EOIR.py"]