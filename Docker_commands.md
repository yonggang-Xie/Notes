# View docker

``` docker images ```

``` docker ps -a ```

# Create a docker container from docker image.

```
docker run -it --gpus=all --name container -v <host_dir>:<docker_dir> image:latest /bin/bash
```

```
docker run -dit  --gpus=all -v /home/xyq1896/workplace/:/workplace/ --name my_dock  pytorch/pytorch
```

- image:latest, the name of docker image
- <host_dir>:<docker_dir> Use absolute path

# Enter Docker

```
docker exec -it my_dock /bin/bash
```

# Start, Stop, remove

``` docker start my_doc ```

``` docker stop my_doc ```

``` docker rm my_doc ```

