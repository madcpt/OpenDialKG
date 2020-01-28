DIRECTORY="./opendialkg/data/"

if [ ! -d "$DIRECTORY" ]; then
  git clone https://github.com/facebookresearch/opendialkg.git opendialkg/
fi

python3 preprocess/split_dataset.py
python3 preprocess/create_data.py

#mkdir save
#mkdir save/KG
#mkdir save/Dial