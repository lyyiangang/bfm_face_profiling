# -u $(id -u) \
docker run -it --rm --gpus all \
-v ${PWD}:${PWD} \
-w ${PWD} \
bfm_face_profilling:latest