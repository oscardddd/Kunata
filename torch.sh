
echo "Installing Anaconda"
sudo wget -P /data https://repo.anaconda.com/archive/Anaconda3-2024.02-1-Linux-x86_64.sh
sudo chmod +x /data/Anaconda3-2024.02-1-Linux-x86_64.sh
sudo bash /data/Anaconda3-2024.02-1-Linux-x86_64.sh -b -p /data/anaconda3
export PATH="/data/anaconda3/bin:$PATH"
echo 'export PATH="/data/anaconda3/bin:$PATH"' >> ~/.bashrc
sudo chmod a+rw -R /data/anaconda3

echo "Creating conda environment"
conda create --name torch python=3.10 -y
eval "$(conda shell.bash hook)"
conda activate torch

echo "Installing Python packages"
pip3 install --upgrade pip
pip3 install torch
pip3 install numpy