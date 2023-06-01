#!/bin/bash
# Copy this and .server.env to an EC2 instance and run it to launch the Text Recognizer server
set -euo pipefail
IFS=$'\n\t'

if [ -f ~/.server.env ]; then
  # shellcheck disable=SC1090
  . ~/.server.env
fi

SESSION="fsdl-text-recognizer"
PORT=11717

if command -v docker &> /dev/null; then
    echo "Docker is installed."
else
    curl -fsSL https://get.docker.com -o get-docker.sh
    sudo sh get-docker.sh
fi

if command -v aws &> /dev/null; then
    echo "AWS CLI is installed."
else
    sudo apt-get install --assume-yes awscli
fi

if command -v ngrok &> /dev/null; then
    echo "ngrok is installed."
else
    curl -s https://ngrok-agent.s3.amazonaws.com/ngrok.asc | sudo tee /etc/apt/trusted.gpg.d/ngrok.asc >/dev/null && echo "deb https://ngrok-agent.s3.amazonaws.com buster main" | sudo tee /etc/apt/sources.list.d/ngrok.list && sudo apt update && sudo apt-get install --assume-yes ngrok
fi

aws configure set default.region "$AWS_DEFAULT_REGION"
aws configure set aws_access_key_id "$AWS_ACCESS_KEY_ID"
aws configure set aws_secret_access_key  "$AWS_SECRET_ACCESS_KEY"

# authenticate to AWS Elastc Container Registry
aws ecr get-login-password --region "$AWS_DEFAULT_REGION" | sudo docker login --username AWS --password-stdin "$AWS_ACCT_ID".dkr.ecr."$AWS_DEFAULT_REGION".amazonaws.com

# Create a new tmux session
tmux new-session -d -s $SESSION

# Split the window into two panes
tmux split-window -t $SESSION:0 -h

# Run the frontend server in the first pane
tmux send-keys -t "$SESSION":0.0 "sudo docker run -it -p $PORT:11700 --rm --env-file ~/.server.env $AWS_ACCT_ID.dkr.ecr.$AWS_DEFAULT_REGION.amazonaws.com/fsdl-text-recognizer/frontend:latest --port 11700 --flagging --gantry --model_url $LAMBDA_URL" Enter

# Run the ngrok tunnel in the second pane
ngrok config add-authtoken "$NGROK_AUTHTOKEN"
tmux send-keys -t "$SESSION":0.1 "ngrok --region=us --hostname=$NGROK_HOSTNAME.ngrok.io http $PORT" Enter

# Focus on the first pane
tmux select-pane -t "$SESSION":0.0

# Attach to the tmux session
tmux attach-session -t $SESSION
