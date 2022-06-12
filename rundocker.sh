docker run -it --rm --gpus all \
-u $(id -u) \
-v ${PWD}:${PWD} \
-w ${PWD} \
bfm_face_profilling:latest