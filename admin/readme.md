# Text Recognizer Project - Admin Readme

## Tasks

```sh
admin/tasks/subset_repo_for_labs.py  # Creates -in _labs by default

admin/tasks/subset_repo_for_labs.sh # Creates in ../fsdl-text-recognition-project, which should be the public git repo
```

Uploading data to S3 is done with `aws s3 cp data/raw/iam/iamdb.zip s3://fsdl-public-assets/iam/iamdb.zip --profile fsdl`
