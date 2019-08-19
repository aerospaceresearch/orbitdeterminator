#!/bin/sh

##
# This script is to install OrbitDeterminator
##

DIR="${HOME}/orbitdeterminator"

if [ ! -d "$DIR" ];
  then
    git clone https://github.com/aerospaceresearch/orbitdeterminator "$DIR"
elif [ ! -f "${DIR}/main.py" ];
  then
    for FILE in ${DIR}/
      do
         rm -r "${FILE}"
      done
    git clone https://github.com/aerospaceresearch/orbitdeterminator "$DIR"
else
    echo "else"
fi

pip install attrs==17.4.0
pip install colorama==0.3.9
pip install cycler==0.10.0
pip install numpy==1.14.0
pip install pluggy==0.6.0
pip install py==1.5.2
pip install pyparsing==2.2.0
pip install pytest==3.4.0
pip install python-dateutil==2.6.1
pip install pytz==2018.3
pip install scipy==1.0.0
pip install sgp4==1.4
pip install six==1.11.0
pip install beautifulsoup4==4.6
pip install https://cdn.mysql.com/Downloads/Connector-Python/mysql-connector-python-1.0.12.tar.gz
pip install requests
pip install inquirer
pip install astropy

export PATH="$PATH:$HOME/bin"
ln -s -f $HOME/orbitdeterminator/orbitdeterminator/main.py $HOME/bin/orbitdeterminator
chmod +x "$HOME/orbitdeterminator/orbitdeterminator/main.py"