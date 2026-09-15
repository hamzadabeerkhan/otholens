# Decision 0002 Container first development

Status: accepted for MVP v0.1

Docker Compose is the supported development and demonstration interface. CPU execution is the baseline. GPU support will be an optional profile when a trained model requires it.

Datasets, secrets, source medical images, and restricted weights must remain outside container images and Git. Containers must run as non-root users and expose health checks.

