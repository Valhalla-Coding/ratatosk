# Rolling back to LM Studio and ComfyUI on their own

Use this if Ratatosk misbehaves and you want the old setup back in a minute. SillyTavern, ComfyCharacters and
other clients need no change, because Ratatosk took over the same ports (11434 and 8188).

1. Stop Ratatosk, as user `master` on Idun:
   `systemctl --user stop ratatosk` (add `systemctl --user disable ratatosk` to keep it from starting again).
   This frees ports 11434 and 8188 and stops its `llama-server`.
2. ComfyUI: in Stability Matrix, put its launch options back to what they were before the cutover:
   `--preview-method auto --use-pytorch-cross-attention --enable-manager --listen`
   (during the cutover they are `... --listen 127.0.0.1 --port 8189`). Restart ComfyUI.
3. LM Studio (if it is still installed): Developer tab, start the server on port 11434.
   If LM Studio was already removed, start the engine by hand instead:
   `LD_LIBRARY_PATH=~/ratatosk/engine/backend:~/ratatosk/engine/vendor ~/ratatosk/engine/backend/llama-server --model <file.gguf> --ctx-size 16384 --host 0.0.0.0 --port 11434 --fit on --flash-attn on --parallel 1`
4. Check: `curl localhost:11434/v1/models` and open `http://192.168.1.11:8188`.
