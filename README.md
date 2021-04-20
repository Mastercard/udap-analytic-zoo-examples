# udap_ncf_analytic_zoo_examples
A demo project elaborate how to use intel analytic zoo to train and inference a NCF deep learning model 
## How to build it.

### By default, build docker image:

    sudo docker build --rm -t ncf_analytic_zoo:latest .
	
### You can also start the container first

    sudo docker run -it --rm  \
    -p 22:22 -p 4040:4040 -p 6006:6006 -p 6379:6379 -p 8080:8080 -p 8998:8998 -p 12345:12345 \
    -e NotebookPort=12345 \
    -e NotebookToken="mc123" \
    -e RUNTIME_DRIVER_CORES=1 \
    -e RUNTIME_DRIVER_MEMORY=2g \
    -e RUNTIME_EXECUTOR_CORES=2 \
    -e RUNTIME_EXECUTOR_MEMORY=6g \
    -e RUNTIME_TOTAL_EXECUTOR_CORES=2 \
    --name ncf_analytic_zoo -h ncf_analytic_zoo ncf_analytic_zoo:latest bash

## How to run the demo code.

### Move to /opt/work/ncf folder and execute ./run_ncf.sh

    cd /opt/work/examples/ncf 
    ./run_ncf.sh

### Start notebook 
    cd /opt/work/examples/ncf 
    unzip jobs.zip 
    cd /opt/work/scripts 
    ./start-notebook.sh 

### Then log in to run the ncf_zoo.ipynb , remember to replace 10.157.146.29 with your host IP address 

   http://10.157.146.29:12345/?token=mc123
