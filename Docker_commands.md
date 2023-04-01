# View docker

``` docker images ```

``` docker ps -a ```

# Create a docker container from docker image.

```
docker run -dit -v /home/xyq1896/workplace/:/workplace/ --name my_dock  pytorch/pytorch
```

- pytorch/pytorch is the name of docker image
- <host_dir>:<docker_dir> Use absolute path

# Enter Docker

```
docker exec -it my_dock /bin/bash
```

# Start, Stop, remove

``` docker start my_doc ```

``` docker stop my_doc ```

``` docker rm my_doc ```

