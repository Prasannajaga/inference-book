# Local Kubernetes Lab — Installation

Use this guide once to prepare your Linux machine for the local Kubernetes lab. The lab itself is in [localsetup-kuberenetes.md](localsetup-kuberenetes.md).

## What needs to be installed

- **Docker Engine** runs the local Kubernetes nodes as containers.
- **kubectl** is the command-line tool used to manage Kubernetes.
- **kind** creates Kubernetes clusters inside Docker.
- **curl** downloads tools and makes HTTP test requests.

## 1. Check your system

```bash
uname -a
uname -m
cat /etc/os-release
```

These commands show your Linux version and CPU architecture. The installation commands below support `x86_64` (AMD/Intel 64-bit) and `aarch64`/`arm64` (64-bit ARM).

## 2. Install and prepare Docker

Install Docker Engine for your Linux distribution first, following the official Docker documentation for that distribution. Then verify and prepare it:

```bash
docker --version
docker info
sudo systemctl status docker
```

These confirm that Docker is installed and its background service (the Docker daemon) is running.

If Docker is stopped, start it and make it start after a reboot:

```bash
sudo systemctl start docker
sudo systemctl enable docker
```

Allow your current user to run Docker commands without `sudo`:

```bash
sudo usermod -aG docker $USER
newgrp docker
```

`newgrp docker` refreshes group membership in the current shell. Signing out and back in also works.

Test Docker:

```bash
docker run --rm hello-world
docker ps
```

The first command downloads and runs a small test container, then removes it. The second lists any currently running containers.

## 3. Install required utilities

```bash
sudo apt update
sudo apt install -y curl ca-certificates
curl --version
```

This refreshes the APT package list, installs `curl` and trusted certificate files, and confirms that `curl` works.

## 4. Install kubectl

Choose the correct download architecture:

```bash
ARCH=$(uname -m)

case "$ARCH" in
  x86_64) KARCH=amd64 ;;
  aarch64|arm64) KARCH=arm64 ;;
  *)
    echo "Unsupported architecture: $ARCH"
    exit 1
    ;;
esac
```

Download the current stable `kubectl`, install it for all users, and verify it:

```bash
KUBECTL_VERSION=$(curl -L -s https://dl.k8s.io/release/stable.txt)
echo "$KUBECTL_VERSION"
curl -LO "https://dl.k8s.io/release/${KUBECTL_VERSION}/bin/linux/${KARCH}/kubectl"
sudo install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl
rm kubectl
kubectl version --client
```

`kubectl` is the tool you will use to create, inspect, and change Kubernetes resources.

## 5. Install kind

Choose the correct download architecture:

```bash
ARCH=$(uname -m)

if [ "$ARCH" = "x86_64" ]; then
  KIND_ARCH="amd64"
elif [ "$ARCH" = "aarch64" ] || [ "$ARCH" = "arm64" ]; then
  KIND_ARCH="arm64"
else
  echo "Unsupported architecture: $ARCH"
  exit 1
fi
```

Download, install, and verify kind:

```bash
curl -Lo ./kind "https://kind.sigs.k8s.io/dl/v0.32.0/kind-linux-${KIND_ARCH}"
chmod +x ./kind
sudo mv ./kind /usr/local/bin/kind
kind version
```

`kind` stands for **Kubernetes in Docker**. It will create your local Kubernetes cluster using Docker containers.

## Ready for the lab

Run these final checks:

```bash
docker run --rm hello-world
kubectl version --client
kind version
```

When all three work, continue with [the local Kubernetes lab](localsetup-kuberenetes.md).
