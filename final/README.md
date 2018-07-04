##

README
===========================
anaconda 環境python使用套件版本
--------------------------
* absl-py                   0.1.13
* astor                     0.6.2
* astroid                   1.6.3      
* bleach                    1.5.0      
* boto                      2.48.0    
* boto3                     1.7.13          
* botocore                  1.10.13         
* bz2file                   0.98            
* ca-certificates           2017.08.26      
* certifi                   2018.1.18       
* chardet                   3.0.4           
* cycler                    0.10.0          
* decorator                 4.2.1      
* docutils                  0.14         
* enum34                    1.1.6        
* future                    0.16.0         
* gast                      0.2.0          
* __gensim__                    __3.1.0__         
* grpcio                    1.10.0         
* __h5py__                      __2.7.1__          
* __html5lib__                  __0.9999999__      
* idna                      2.6            
* intel-openmp              2018.0.0       
* isort                     4.3.4          
* __jieba__                     __0.39__           
* jmespath                  0.9.3          
* __Keras__                     __2.0.8__          
* __keras-vis__                 __0.4.1__          
* kiwisolver                1.0.1          
* lazy-object-proxy         1.3.1          
* libgcc-ng                 7.2.0          
* libgfortran-ng            7.2.0          
* libstdcxx-ng              7.2.0          
* Markdown                  2.6.11         
* __matplotlib__                __2.2.0__          
* mccabe                    0.6.1          
* mkl                       2018.0.1       
* natsort                   5.2.0          
* networkx                  2.1            
* __numpy__                     __1.13.0__         
* olefile                   0.45.1         
* openssl                   1.0.2n         
* pandas                    0.21.1        
* parsedatetime             2.4           
* __Pillow__                    __5.0.0__
* pip                       9.0.3         
* pip                       9.0.1         
* protobuf                  3.5.2.post1   
* pylint                    1.8.4         
* pyparsing                 2.2.0        
* __python__                    __3.5.2__
* python-dateutil           2.6.1        
* python-utils              2.3.0        
* pytz                      2018.3           
* PyWavelets                0.5.2            
* PyYAML                    3.12       
* readline                  6.2        
* recurrent                 0.2.5      
* requests                  2.18.4     
* s3transfer                0.1.13     
* __scikit-image__              __0.13.1__     
* __scikit-learn__              __0.19.1__     
* __scipy__                     __0.19.1__     
* setuptools                39.0.1     
* setuptools                38.5.1     
* six                       1.11.0     
* __sklearn__                   __0.19.1__     
* smart-open                1.5.7      
* sqlite                    3.13.0     
* __tensorboard__               __1.6.0__      
* __tensorflow-gpu__            __1.4.0__      
* __tensorflow-tensorboard__    __0.4.0__      
* termcolor                 1.1.0      
* tk                        8.5.18     
* torch                     0.3.0      
* torchvision               0.2.0      
* urllib3                   1.22       
* Werkzeug                  0.14.1     
* wheel                     0.30.0     
* wrapt                     1.10.11    
* xz                        5.2.3      
* zlib                      1.2.11     

使用方式：
----------------------------
kaggle reproduce：
```Bash
bash predict.sh <testing_data_file_dir> <output_file>
```
example:
```Bash
bash predict.sh ./data/testing_data.csv ./reproduce.csv
```
檢查是否與kaggle檔案一致
```Bash
bash checkreproduce.sh <testing_file>
bash checkreproduce.sh ./reproduce.csv
```
PCA預測：
```Bash
cd PCA_model
bash pcapre.sh <training_data_file_dir> <testing_data_file> <outputfile>
```
example:
```Bash
bash pcapre.sh ../data/training_data/ ../data/testing_data.csv ./output.csv
```
備註：
training_data_file_dir為放置 1_train.txt~5_train.txt之資料夾路徑
