#!/bin/bash
cd src
python3 ./model_1/pre.py '../'$1 ./result/s0.csv ./model_1/s0.h5
python3 ./model_1/pre.py '../'$1 ./result/s1.csv ./model_1/s1.h5
python3 ./model_1/pre.py '../'$1 ./result/s2.csv ./model_1/s2.h5
python3 ./model_1/pre.py '../'$1 ./result/s3.csv ./model_1/s3.h5
python3 ./model_1/pre.py '../'$1 ./result/s4.csv ./model_1/s4.h5
if [ -f "./model_2/ver0.h5" ]; then
    echo "File ./model_2/ver0.h5 exists, skip download."
else
    wget -O ./model_2/ver0.h5 https://www.dropbox.com/s/2pbkbh3ngz0mlsj/ver0.h5
fi
python3 ./model_2/pre.py '../'$1 ./result/ver0.csv ./model_2/ver0.h5
if [ -f "./model_2/ver1.h5" ]; then
    echo "File ./model_2/ver1.h5 exists, skip download."
else
    wget -O ./model_2/ver1.h5 https://www.dropbox.com/s/7xhotjelu9dvonm/ver1.h5
fi
python3 ./model_2/pre.py '../'$1 ./result/ver1.csv ./model_2/ver1.h5
if [ -f "./model_2/ver2.h5" ]; then
    echo "File ./model_2/ver2.h5 exists, skip download."
else
    wget -O ./model_2/ver2.h5 https://www.dropbox.com/s/8wpslayaxua9m16/ver2.h5
fi
python3 ./model_2/pre.py '../'$1 ./result/ver2.csv ./model_2/ver2.h5
if [ -f "./model_2/ver5.h5" ]; then
    echo "File ./model_2/ver5.h5 exists, skip download."
else
    wget -O ./model_2/ver5.h5 https://www.dropbox.com/s/y5b3yu4q4hjqm9r/ver5.h5
fi
python3 ./model_2/pre.py '../'$1 ./result/ver5.csv ./model_2/ver5.h5
if [ -f "./model_2/ver6.h5" ]; then
    echo "File ./model_2/ver6.h5 exists, skip download."
else
    wget -O ./model_2/ver6.h5 https://www.dropbox.com/s/w9f5mb3rsv9kgub/ver6.h5
fi
python3 ./model_2/pre.py '../'$1 ./result/ver6.csv ./model_2/ver6.h5
python3 ./model_4/pre.py '../'$1 ./result/over0.csv ./model_4/over0.h5
python3 ./model_4/pre.py '../'$1 ./result/over1.csv ./model_4/over1.h5
python3 ./model_4/pre.py '../'$1 ./result/over2.csv ./model_4/over2.h5
python3 ./model_4/pre.py '../'$1 ./result/over3.csv ./model_4/over3.h5
python3 ./model_4/pre.py '../'$1 ./result/over4.csv ./model_4/over4.h5
python3 ./model_5/pre.py '../'$1 ./result/150over3.csv ./model_5/150over3.h5
python3 ./model_5/pre.py '../'$1 ./result/150over4.csv ./model_5/150over4.h5
if [ -f "./model_6/w2v" ]; then
    echo "File ./model_6/w2v exists, skip download."
else
    wget -O ./model_6/w2v https://www.dropbox.com/s/ohd3eb7r9mfnas6/w2v
fi
python3 ./model_6/pre.py '../'$1 ./result/125s3.csv ./model_6/125s3.h5
python3 ./model_6/pre.py '../'$1 ./result/125s4.csv ./model_6/125s4.h5
python3 ./other/pre_300s.py '../'$1 ./result/300s.csv ./other/300s.h5
python3 ./other/pre_200.py '../'$1 ./result/200.csv ./other/200.h5
cd ..
python3 ./src/result/ensemble.py $2
