- `sudo swapoff -a`
- `sudo sed -i '/ swap / s/^/#/' /etc/fstab`
- `sudo hostnamectl set-hostname "master-node"`
- `exec bash`

## set host names

- `sudo nano /etc/hosts`

## Set up the IPV4 bridge on all nodes

cat <<EOF | sudo tee /etc/modules-load.d/k8s.conf
overlay
br_netfilter
EOF

- sudo modprobe overlay
- sudo modprobe br_netfilter

# sysctl params required by setup, params persist across reboots
cat <<EOF | sudo tee /etc/sysctl.d/k8s.conf
net.bridge.bridge-nf-call-iptables  = 1
net.bridge.bridge-nf-call-ip6tables = 1
net.ipv4.ip_forward                 = 1
EOF

# Apply sysctl params without reboot
- `sudo sysctl --system`

## Install kubelet, kubeadm, and kubectl on each node
- `sudo apt-get update`
- `sudo apt-get install -y apt-transport-https ca-certificates curl`
- `sudo mkdir /etc/apt/keyrings`
- `curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo gpg --dearmor -o /etc/apt/keyrings/kubernetes-archive-keyring.gpg`
- `echo "deb [signed-by=/etc/apt/keyrings/kubernetes-archive-keyring.gpg] https://apt.kubernetes.io/ kubernetes-xenial main" | sudo tee /etc/apt/sources.list.d/kubernetes.list`
- `sudo apt-add-repository "deb http://apt.kubernetes.io/ kubernetes-xenial main"`
- `sudo apt-get update`
- `sudo apt-get install -y kubelet kubeadm kubectl`
## install docker and containerd
 - `sudo apt install docker.io`
- `sudo mkdir /etc/containerd`
- `sudo sh -c "containerd config default > /etc/containerd/config.toml"`
- `sudo sed -i 's/ SystemdCgroup = false/ SystemdCgroup = true/' /etc/containerd/config.toml`
- `sudo systemctl restart containerd.service`
- `sudo systemctl restart kubelet.service`
- `sudo systemctl enable kubelet.service`

## start cluster on master
- `sudo kubeadm config images pull`
- `sudo kubeadm init --pod-network-cidr=10.0.0.0/16`
- `mkdir -p $HOME/.kube`
- `sudo cp -i /etc/kubernetes/admin.conf $HOME/.kube/config`
- `sudo chown $(id -u):$(id -g) $HOME/.kube/config`
- `kubectl create -f https://raw.githubusercontent.com/projectcalico/calico/v3.26.1/manifests/tigera-operator.yaml`
- `curl https://raw.githubusercontent.com/projectcalico/calico/v3.26.1/manifests/custom-resources.yaml -O`
- `sed -i 's/cidr: 192\.168\.0\.0\/16/cidr: 10.10.0.0\/16/g' custom-resources.yaml`
- `kubectl create -f custom-resources.yaml`
## add worker nodes

## get nodes
- `kubectl get no`
- `kubectl get po -A`