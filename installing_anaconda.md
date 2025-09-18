# installing conda
``bash Anaconda3-2024.10-1-Linux-x86_64.sh -b -p /aiau001_scratch/anaconda3``
# export conda
``echo 'export PATH="/aiau001_scratch/tin/anaconda3/bin:$PATH"' >> ~/.bashrc``

``source ~/.bashrc``

# create new env
``conda create -n myenv python=3.10``

# init conda
``/aiau001_scratch/tin/anaconda3/bin/conda init bash``
