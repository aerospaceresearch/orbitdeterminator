#!/bin/sh

##
# This script is to install DirectDemod
##

## variables
DIR="${HOME}/orbitdeterminator"

if [ ! -d "$DIR" ];
  then
    git clone https://github.com/aerospaceresearch/orbitdeterminator "$DIR"
elif [ ! -f "${DIR}/main.py" ];
  then
    for FILE in ${DIR}/
      do
        sudo rm -r "${FILE}"
      done
    git clone https://github.com/aerospaceresearch/orbitdeterminator "$DIR"
else
    echo "else"
fi

sudo conda create -n aero python=3.5.2
sudo conda activate aero
sudo pip install attrs==17.4.0
sudo pip install colorama==0.3.9
sudo pip install cycler==0.10.0
sudo pip install numpy==1.14.0
sudo pip install pluggy==0.6.0
sudo pip install py==1.5.2
sudo pip install pyparsing==2.2.0
sudo pip install pytest==3.4.0
sudo pip install python-dateutil==2.6.1
sudo pip install pytz==2018.3
sudo pip install scipy==1.0.0
sudo pip install sgp4==1.4
sudo pip install six==1.11.0
sudo pip install beautifulsoup4==4.6
sudo pip install https://cdn.mysql.com/Downloads/Connector-Python/mysql-connector-python-1.0.12.tar.gz
sudo pip install requests
sudo pip install inquirer
sudo pip install astropy
sudo conda install pykep=2.1
sudo conda install matplotlib

export PATH="$PATH:$HOME/bin"
sudo ln -s $HOME/orbitdeterminator/orbitdeterminator/main.py $HOME/bin/orbitdeterminator
chmod +x "$DIR/orbitdeterminator/main.py"

printf "\nAll task completed, use the program with 'orbitdeterminator' command\n"