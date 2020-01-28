# make directories
if [ ! -d "./save" ]; then
  mkdir ./save
fi

if [ ! -d "./save/KGE" ]; then
  mkdir ./save/KGE
fi

python3 ./KGE/TransE.py -b=256