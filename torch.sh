
echo "Installing Anaconda"
sudo wget -P ../ https://repo.anaconda.com/archive/Anaconda3-2024.02-1-Linux-x86_64.sh
sudo chmod +x ./Anaconda3-2024.02-1-Linux-x86_64.sh
sudo bash ../Anaconda3-2024.02-1-Linux-x86_64.sh -b -p ../anaconda3
export PATH="../anaconda3/bin:$PATH"
echo 'export PATH="./anaconda3/bin:$PATH"' >> ~/.bashrc
sudo chmod a+rw -R ./anaconda3

echo "Creating conda environment"
conda create --name torch python=3.10 -y
eval "$(conda shell.bash hook)"
conda activate torch24

echo "Installing Pytorch 2.4.1"
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cpu
pip3 install numpy