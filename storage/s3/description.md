Purpose: Interface to S3/Minio for raw data storage.

Expected contents:
- S3/Minio client implementation
- Upload/download utilities
- Path builders (patient → folder → artifacts)
- Optional:
    - presigned URL generators
    - retry logic
    - integrity checks

This folder handles storing raw uploaded patient data and linking it to DB entries.