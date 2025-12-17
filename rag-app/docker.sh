podman build --platform linux/amd64 -t rag-app .
podman tag rag-app docker.io/arnab135/rag-app:latest
podman push docker.io/arnab135/rag-app:latest