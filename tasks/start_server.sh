#!/bin/bash
# Copy this and .server.env to an EC2 instance and run it to launch the Text Recognizer server

if [ -f ~/.server.env ]; then
  # shellcheck disable=SC1090
  . ~/.server.env
fi

SESSION="fsdl-text-recognizer"
PORT=11717

# Create a new tmux session
tmux new-session -d -s $SESSION

# Split the window into two panes
tmux split-window -t $SESSION:0 -h

# Run the frontend server in the first pane
tmux send-keys -t "$SESSION":0.0 "sudo docker run -it -p $PORT:11700 --rm --env-file .env $AWS_ACCT_ID.dkr.ecr.$REGION.amazonaws.com/fsdl-text-recognizer/frontend:latest --port 11700 --flagging --gantry --model_url $LAMBDA_URL" Enter

# Run the ngrok tunnel in the second pane
tmux send-keys -t "$SESSION":0.1 "ngrok --region=us --hostname=fsdl-text-recognizer.ngrok.io http $PORT" Enter

# Focus on the first pane
tmux select-pane -t "$SESSION":0.0

# Attach to the tmux session
tmux attach-session -t $SESSION
