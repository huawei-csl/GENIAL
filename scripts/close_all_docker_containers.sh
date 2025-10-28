#!/bin/bash

docker container stop $(docker ps --filter status=running --filter name=launch_script -q)
docker rm -v $(docker ps --filter status=exited --filter name=launch_script -q)
docker rm -v $(docker ps --filter status=created --filter name=launch_script -q)

