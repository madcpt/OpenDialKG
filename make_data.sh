DIRECTORY="./opendialkg/data/"

if [ ! -d "$DIRECTORY" ]; then
  git clone https://github.com/facebookresearch/opendialkg.git opendialkg/
fi

python3 create_data.py
