#!/bin/sh

USER=root

byobu new-session -d -s $USER

# Train windows
byobu rename-window -t $USER:0 'train'
#byobu send-keys "<>" C-m

byobu new-window -t $USER:1 -n 'visdom'
byobu send-keys "python -m visdom.server -env_path out/runs/visdom/ -port 6006" C-m

byobu new-window -t $USER:2 -n 'books'
byobu send-keys "jupyter notebook --port 8080 --allow-root --ip 0.0.0.0 --no-browser" C-m

# Set default window as the dev split plane
byobu select-window -t $USER:0

byobu split-window -h
byobu send-keys "watch nvidia-smi" C-m
byobu split-window -v
byobu send-keys "htop" C-m

# Attach to the session you just created
# (flip between windows with alt -left and right)
byobu attach-session -t $USER
