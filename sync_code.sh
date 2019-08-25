#!/bin/bash

: <<'END'

rsync commands:
    a: archive mode - rescursive, preserves owner, preserves permissions, preserves modification times, preserves group, copies symlinks as symlinks, preserves device files.
    H: preserves hard-links
    A: preserves ACLs
    X: preserves extended attributes
    x: don't cross file-system boundaries
    v: increase verbosity
    --numeric-ds: don't map uid/gid values by user/group name
    --delete: delete extraneous files from dest dirs (differential clean-up during sync)
    --progress: show progress during transfer

ssh commands:
    T: turn off pseudo-tty to decrease cpu load on destination.
    o Compression=no: Turn off SSH compression.
    x: turn off X forwarding if it is on by default.

END

# argument sync from local to remote OR remote to local
echo $1
if [ $1 = "r2l" ]
then
    echo "====================== SYNC FROM REMOTE TO LOCAL ======================"
    rsync -aHAXxv --progress -e "ssh -T -o Compression=no -x" \
    production-ml:/home/ubuntu/mays_workspace/.wheelchair_commands/ .
elif [ $1 = "l2r" ]
then
    echo "====================== SYNC FROM LOCAL TO REMOTE ======================"
    rsync -aHAXxv --numeric-ids --exclude omniglot --exclude .idea --exclude .ipynb_checkpoints --exclude __pycache__ --exclude experiment_results.ipynb --exclude models --exclude .git --exclude .vscode --exclude .DS_Store --exclude .npy --progress -e  "ssh -T -o Compression=no -x" \
    . production-ml:/home/ubuntu/mays_workspace/.wheelchair_commands/
else
    echo "====================== COMMANDS LIST ======================"
    echo "====================== r2l -> sync from remote to local ======================"
    echo "====================== l2r -> sync from local to remote ======================"
fi
