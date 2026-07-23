# Local Kubernetes Lab

This guide walks you through setting up a local Kubernetes cluster to test and verify Kubernetes concepts live on your machine.

Instead of a generic web server, we will build and deploy a **FastAPI microservice** that exposes a `/health` endpoint, package it into a container, deploy it across multiple worker nodes, expose it via a Kubernetes `Service`, set up **NGINX Ingress Controller** for direct external domain routing (no `port-forward` needed), and test how `kube-apiserver`, `kube-controller-manager`, `kube-scheduler`, `kubelet`, `kube-proxy`, `EndpointSlices`, and health probes operate under the hood.

Before starting, make sure you have completed the [installation guide](install.md).

---

## 1. Create a Workspace

```bash
mkdir -p ~/k8s-test-cluster
cd ~/k8s-test-cluster
pwd
```

**What this creates:** A workspace directory on your machine to hold your application code, Dockerfiles, and Kubernetes manifests (`kind-cluster.yaml`, `deployment.yaml`, `service.yaml`, `ingress.yaml`).

---

## 2. Build the FastAPI Container Application

### Step 2.1: Write the Application Code

```bash
cat << 'EOF' > main.py
import os
import socket
from fastapi import FastAPI, Response, status

app = FastAPI(title="K8s Warehouse Robot")

@app.get("/")
def read_root():
    return {
        "message": "Hello from the Kubernetes Warehouse Robot!",
        "pod_name": socket.gethostname(),
        "pod_ip": os.getenv("POD_IP", "unknown"),
        "node_name": os.getenv("NODE_NAME", "unknown")
    }

@app.get("/health")
def health_check(response: Response):
    # Returns HTTP 200 OK with health status and current Pod name
    return {
        "status": "healthy",
        "pod": socket.gethostname()
    }
EOF
```

**What this creates:** `main.py` — the application code running inside your Pod containers. It includes:

- `GET /health`: The HTTP endpoint queried by `kubelet` for readiness and liveness probes.
- `GET /`: Returns the current Pod name, Node name, and Pod IP to verify load balancing across different worker nodes.

---

### Step 2.2: Write the Dockerfile

```bash
cat << 'EOF' > Dockerfile
FROM python:3.11-slim
WORKDIR /app
RUN pip install --no-cache-dir fastapi uvicorn
COPY main.py .
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
EOF
```

**What this creates:** `Dockerfile` — the container specification instructing the container runtime (`containerd`) how to package Python, install dependencies, and run FastAPI on port `8000`.

---

### Step 2.3: Build the Local Docker Image

```bash
docker build -t fastapi-app:v1 .
docker images | grep fastapi-app
```

**What this creates:** A compiled container image named `fastapi-app:v1`. This image serves as the template that Kubernetes worker nodes use to instantiate Pod containers.

---

## 3. Create the Multi-Node Cluster (`kind`)

### Step 3.1: Define Cluster Topology with Host Port Mappings

```bash
cat << 'EOF' > kind-cluster.yaml
kind: Cluster
apiVersion: kind.x-k8s.io/v1alpha4

nodes:
  - role: control-plane
    kubeadmConfigPatches:
      - |
        kind: InitConfiguration
        nodeRegistration:
          kubeletExtraArgs:
            node-labels: "ingress-ready=true"
    extraPortMappings:
      - containerPort: 80
        hostPort: 80
        protocol: TCP
      - containerPort: 443
        hostPort: 443
        protocol: TCP
  - role: worker
  - role: worker
  - role: worker
EOF
```

**What this creates:** `kind-cluster.yaml` defining:

- **1 Control-Plane Node**: Houses control plane components (`kube-apiserver`, `etcd`, `kube-controller-manager`, `kube-scheduler`). Includes `extraPortMappings` linking host ports `80` and `443` directly to the cluster control-plane node.
- **3 Worker Nodes**: Defines 3 separate worker nodes where node agents (`kubelet`, `kube-proxy`, `containerd`) run workloads.

---

### Step 3.2: Spin Up the Cluster

```bash
kind create cluster \
  --name k8s-test-cluster \
  --config kind-cluster.yaml \
  --wait 5m
```

**What this creates:** A multi-node Kubernetes cluster by launching 4 Docker containers as virtual nodes. It initializes the Control Plane and 3 Worker Nodes with host port 80 forwarding enabled.

---

### Step 3.3: Load Image into Worker Nodes

```bash
kind load docker-image fastapi-app:v1 --name k8s-test-cluster
```

**What this creates:** Pre-loads `fastapi-app:v1` directly into the container runtime (`containerd`) image cache on all 3 worker nodes, bypassing the need for a remote container registry.

---

### Step 3.4: Verify Cluster Architecture

```bash
kubectl cluster-info
kubectl get nodes -o wide
kubectl get pods -n kube-system
```

**What this verifies:** Queries `kube-apiserver` to verify that:

1. The Control Plane components (`kube-apiserver`, `etcd`, `kube-scheduler`, `kube-controller-manager`) are active.
2. All 3 Worker Nodes are in `Ready` status with running `kubelet` and `kube-proxy` agents.

---

## 4. Deploy the FastAPI Application

### Step 4.1: Write the Deployment Manifest

```bash
cat << 'EOF' > deployment.yaml
apiVersion: apps/v1  # API group and version for Deployment controller
kind: Deployment     # Resource type that manages ReplicaSets and Pods
metadata:
  name: fastapi-app   # Name of this deployment object in Kubernetes
  namespace: default  # Namespace where this deployment will be created
  labels:
    app: fastapi-robot # Metadata label attached to the deployment object

spec:
  replicas: 3 # Number of identical Pod instances kube-controller-manager maintains

  selector:
    matchLabels:
      app: fastapi-robot # Selector rule telling ReplicaSet which Pods to manage

  template: # Blueprint spec used by ReplicaSet to create each new Pod
    metadata:
      labels:
        app: fastapi-robot # Label assigned to every Pod created from this template
    spec:
      containers:
        - name: fastapi-container    # Name of the container running inside the Pod
          image: fastapi-app:v1      # Local container image name built with Docker
          imagePullPolicy: IfNotPresent # Use local image if available, avoid remote pull
          ports:
            - containerPort: 8000    # Port the FastAPI application listens on inside container

          # Downward API passing Kubernetes Pod runtime metadata to container environment variables
          env:
            - name: POD_NAME
              valueFrom:
                fieldRef:
                  fieldPath: metadata.name # Injects unique Pod name
            - name: POD_IP
              valueFrom:
                fieldRef:
                  fieldPath: status.podIP # Injects assigned internal Pod IP address
            - name: NODE_NAME
              valueFrom:
                fieldRef:
                  fieldPath: spec.nodeName # Injects host worker node name running this Pod

          # Readiness probe checking if container is ready to accept Service traffic
          readinessProbe:
            httpGet:
              path: /health # Endpoint kubelet checks for HTTP 200 OK response
              port: 8000   # Container port for health check
            initialDelaySeconds: 3 # Seconds kubelet waits after container start before first check
            periodSeconds: 5      # Frequency in seconds for subsequent readiness checks

          # Liveness probe checking if container is alive or needs a restart
          livenessProbe:
            httpGet:
              path: /health # Endpoint kubelet checks to verify container responsiveness
              port: 8000   # Container port for liveness check
            initialDelaySeconds: 5 # Seconds kubelet waits before starting liveness checks
            periodSeconds: 10     # Frequency in seconds for subsequent liveness checks

          # Pod machine specs and resource limits enforced by kube-scheduler and Linux kernel
          resources:
            requests:
              cpu: "100m"       # 100 millicores (0.1 CPU core) guaranteed minimum CPU reserved
              memory: "128Mi"   # 128 Mebibytes RAM guaranteed minimum memory reserved
            limits:
              cpu: "300m"       # 300 millicores (0.3 CPU core) maximum CPU limit (throttled if exceeded)
              memory: "256Mi"   # 256 Mebibytes RAM maximum memory limit (terminated if exceeded)
EOF
```

**What this creates:** `deployment.yaml` — the desired state manifest specifying target replica count (3 Pods), label selectors, container image, environment variables, readiness/liveness health probes, and CPU/RAM machine specs.

---

### Step 4.2: Apply Deployment & Watch Lifecycle

```bash
kubectl apply -f deployment.yaml
kubectl get pods -w
```

**What this triggers:** Executes the Kubernetes control loop:

1. `kubectl` sends the spec to `kube-apiserver`, which saves it in `etcd`.
2. `kube-controller-manager` detects a discrepancy (0 running vs 3 desired) and creates a `ReplicaSet`.
3. `kube-scheduler` filters and scores worker nodes, then assigns each Pod to a node.
4. `kubelet` on each node reads its assigned Pod spec and commands `containerd` to run the container.

---

### Step 4.3: Verify Deployment State

```bash
kubectl get deployments
kubectl get replicasets
kubectl get pods -o wide
```

**What this checks:** Inspects cluster state stored in `etcd` to confirm that 3 Pods are running across your worker nodes with unique Pod IPs.

---

## 5. Create a Service & Inspect EndpointSlices

### Step 5.1: Create Service Manifest

```bash
cat << 'EOF' > service.yaml
apiVersion: v1
kind: Service
metadata:
  name: fastapi-service
  namespace: default
spec:
  # Permanent virtual IP inside the cluster
  type: ClusterIP

  # Selector badge rule: match Pods wearing app: fastapi-robot
  selector:
    app: fastapi-robot

  # Port mapping: Service listens on port 80, routes to container port 8000
  ports:
    - name: http
      protocol: TCP
      port: 80
      targetPort: 8000
EOF
```

**What this creates:** `service.yaml` — defining a permanent internal virtual IP (`ClusterIP`) and DNS entry (`fastapi-service`) to route traffic reliably regardless of Pod restarts.

---

### Step 5.2: Apply Service Manifest

```bash
kubectl apply -f service.yaml
kubectl get svc fastapi-service
```

**What this creates:** Registers the Service with `kube-apiserver`. On every node, `kube-proxy` configures internal networking rules (iptables/IPVS) to balance traffic sent to port `80` across healthy Pods on port `8000`.

---

### Step 5.3: Inspect EndpointSlices

```bash
kubectl get endpoints fastapi-service
kubectl get endpointslices -l kubernetes.io/service-name=fastapi-service
kubectl get endpointslices -l kubernetes.io/service-name=fastapi-service -o yaml
```

**What this inspects:** Displays `EndpointSlices` resources, which track the live list of healthy Pod IP addresses that passed the `/health` readiness check.

---

## 6. Deploy NGINX Ingress Controller & Configure Routing

### Step 6.1: Deploy NGINX Ingress Controller Manifest for `kind`

```bash
kubectl apply -f https://raw.githubusercontent.com/kubernetes/ingress-nginx/main/deploy/static/provider/kind/deploy.yaml
```

**What this installs:** Deploys NGINX Ingress Controller resources specifically configured for `kind`.

---

### Step 6.2: Pin Ingress Controller to Control-Plane Node

To ensure NGINX listens directly on host port 80 (mapped via Docker to the `control-plane` container), pin the deployment to `k8s-test-cluster-control-plane`:

```bash
kubectl patch deployment ingress-nginx-controller -n ingress-nginx \
  -p '{"spec":{"template":{"spec":{"nodeName":"k8s-test-cluster-control-plane","tolerations":[{"operator":"Exists"}]}}}}'
```

Wait for NGINX Ingress Controller to reach `1/1 Running`:

```bash
kubectl get pods -n ingress-nginx -o wide -w
```

---

### Step 6.3: Create Ingress Resource (`ingress.yaml`)

```bash
cat << 'EOF' > ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: fastapi-ingress
  namespace: default
spec:
  ingressClassName: nginx
  rules:
    - host: api.localtest.me
      http:
        paths:
          - path: /
            pathType: Prefix
            backend:
              service:
                name: fastapi-service
                port:
                  number: 80
EOF

kubectl apply -f ingress.yaml
```

**What this creates:** `ingress.yaml` configures NGINX Ingress Controller routing rules. Any HTTP request for `api.localtest.me` arriving at port 80 is routed directly to the backend Pod IPs defined in `EndpointSlices`.

---

## 7. Access & Test the Application `/health` Endpoint

### Method A: Direct External Test via Ingress (No Port-Forwarding Needed)

Test directly using your host machine's browser or terminal `curl`:

```bash
# Test main application endpoint via Ingress
curl http://api.localtest.me/

# Test /health endpoint via Ingress
curl http://api.localtest.me/health
```

Expected response from `/health`:

```json
{"status":"healthy","pod":"fastapi-app-xxxxx-yyyyy"}
```

**Traffic Flow:**
Host machine (`http://api.localtest.me`) -> Host port 80 -> NGINX Ingress Controller on Control Plane -> Reads `EndpointSlices` -> `fastapi-app` Pod IP:8000.

---

### Method B: Local Host Testing (Port-Forwarding Fallback)

Why port-forwarding is needed: A `ClusterIP` Service is private and accessible only inside the internal Kubernetes network. Your host machine cannot route traffic directly to `ClusterIP` addresses. `kubectl port-forward` bridges port `8080` on your local machine (`localhost:8080`) directly to port `80` of `fastapi-service` inside the cluster.

Run in Terminal 1:

```bash
kubectl port-forward service/fastapi-service 8080:80
```

Run in Terminal 2:

```bash
# Test main application endpoint
curl http://localhost:8080/

# Test /health endpoint
curl http://localhost:8080/health
```

---

### Method C: In-Cluster Network & DNS Test

```bash
kubectl run curl-test --image=curlimages/curl --restart=Never -it --rm -- sh
```

Inside the `curl-test` prompt:

```sh
# 1. Test via Service short DNS name
curl http://fastapi-service/health

# 2. Test via full Kubernetes FQDN
curl http://fastapi-service.default.svc.cluster.local/health

# 3. Repeat curl multiple times to see load balancing across different Pods
curl http://fastapi-service/
curl http://fastapi-service/
curl http://fastapi-service/

# Exit container
exit
```

---

## 8. Verify Core Component Behavior

### Step 8.1: Inspect Control Plane Containers

```bash
docker exec -it k8s-test-cluster-control-plane crictl ps
```

**What this inspects:** Opens a shell into the Control Plane node container and runs `crictl ps` to view static Pods for `kube-apiserver`, `etcd`, `kube-scheduler`, and `kube-controller-manager`.

---

### Step 8.2: Test Self-Healing (Desired State Loop)

```bash
# Delete one Pod manually
kubectl delete pod <POD_NAME>

# Watch new replacement Pod get created
kubectl get pods -w
```

**What this demonstrates:** Simulates container failure. `kube-controller-manager` compares actual state (2 Pods) with desired state (3 Pods) in `etcd`, detects the gap, and immediately triggers creation of a replacement Pod.

---

### Step 8.3: Test Dynamic Scaling

```bash
kubectl scale deployment fastapi-app --replicas=5
kubectl get pods -o wide
kubectl get endpointslices
```

**What this demonstrates:** Updates desired state to 5 replicas. `kube-scheduler` places 2 new Pods on worker nodes, `kubelet` starts them, and `EndpointSlices` updates with the new Pod IPs automatically. NGINX Ingress Controller receives the updated `EndpointSlices` in real-time.

---

## 9. Clean Up

```bash
kubectl delete -f ingress.yaml
kubectl delete -f service.yaml
kubectl delete -f deployment.yaml
kind delete cluster --name k8s-test-cluster
docker ps
```

**What this does:** Removes application and ingress resources from `etcd` and destroys the 4 Docker node containers, restoring your system state.
