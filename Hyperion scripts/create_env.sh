source /opt/flight/etc/setup.sh
flight env activate gridware
module add gnu
pyenv virtualenv 3.9.5 inm705_CW
echo inm705_CW > Deep-Learning-Img-Analysis/.python-version
mv requirements.txt Deep-Learning-Img-Analysis
cd Deep-Learning-Img-Analysis
which python
python --version
pip3 install --proxy http://hpc-proxy00.city.ac.uk:3128 -r requirements.txt
pip3 install --proxy http://hpc-proxy00.city.ac.uk:3128 torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

