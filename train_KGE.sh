KGE="./save/KGE"

if [ ! -d "./save" ]; then
  mkdir ./save
fi

if [ ! -d "$KGE" ]; then
  mkdir "$KGE"
fi

python3 ./KGE/TransE.py -b=256