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

pip3 install -r "${DIR}/requirements.txt" --user

export PATH="$PATH:$HOME/bin"
ln -s -f $HOME/orbitdeterminator/orbitdeterminator/main.py $HOME/bin/orbitdeterminator
chmod +x "$HOME/orbitdeterminator/orbitdeterminator/main.py"