sudo apt install python3-pip
python3 -m pip install --user --upgrade pip setuptools wheel
python3 -m pip install --user numpy
python3 -m pip install --user scipy
python3 -m pip install --user imutils
python3 -m pip install --user opencv-python
python3 -m pip install --user scikit-image
# Install dlib preriquised
sudo apt-get install build-essential cmake
sudo apt-get install libgtk-3-dev
sudo apt-get install "libgtk-3-dev=3.18.9-1ubuntu3.3"
# Si cette étape échoue, il faut télécharger le package directement :
# 1 - RDV sur https://packages.ubuntu.com/xenial-updates/libgtk-3-dev
# 2 - dpkg --print-architecture pour connaître l'archi système
# 3 - Télécharger l'archive
# 4 - sudo apt -f install ~/Bureau/Elec_Auto/Packages/libgtk-3-dev_3.18.9-1ubuntu3.3_amd64.deb
sudo apt-get install libboost-all-dev

wget https://bootstrap.pypa.io/get-pip.py
sudo python get-pip.py


python3 -m pip install --user dlib


$ pip install numpy
$ pip install scipy
$ pip install scikit-image
	
$ pip install dlib
