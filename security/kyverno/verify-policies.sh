#!/bin/bash

echo "=== Verifying Kyverno policies ==="

echo "Checking policies are installed..."
kubectl get cpol disallow-privileged-kserve
kubectl get cpol require-pod-resources-kserve

echo "Applying test fixtures..."
kubectl apply -f security/kyverno/examples/privileged-pod.yaml
kubectl apply -f security/kyverno/examples/missing-resources-pod.yaml

sleep 5

echo "Checking policy reports..."
kubectl get policyreport -n kserve

echo "Cleaning up test fixtures..."
kubectl delete -f security/kyverno/examples/privileged-pod.yaml
kubectl delete -f security/kyverno/examples/missing-resources-pod.yaml
