
echo "Installing Anaconda"
sudo wget -P ./ https://repo.anaconda.com/archive/Anaconda3-2024.02-1-Linux-x86_64.sh
sudo chmod +x ./Anaconda3-2024.02-1-Linux-x86_64.sh
sudo bash ./Anaconda3-2024.02-1-Linux-x86_64.sh -b -p ./anaconda3
export PATH="./anaconda3/bin:$PATH"
echo 'export PATH="./anaconda3/bin:$PATH"' >> ~/.bashrc
sudo chmod a+rw -R ./anaconda3

echo "Creating conda environment"
conda create --name torch python=3.10 -y
eval "$(conda shell.bash hook)"
conda activate torch

echo "Installing Python packages"
pip3 install torch
pip3 install numpy