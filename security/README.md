# Security Foundations

This folder is the next security phase for the GPON MLOps project.

The important idea is that security should be introduced in layers:

1. Harden the current workloads and add guardrails.
2. Enforce those guardrails with policy.
3. Add identity and zero-trust patterns later.

For this project, the safest order is:

1. `Kyverno` policy enforcement
2. `FastAPI` inference gateway
3. `SPIFFE/SPIRE` workload identity

## Why This Order

### 1. Kyverno first

Kyverno is a good first step because it teaches Kubernetes security in a very visible way.
It lets you say things like:

- every container must have CPU and memory requests/limits
- privileged containers are not allowed
- certain namespaces must follow stricter rules

This gives you immediate value without forcing you to redesign the application.

### 2. FastAPI gateway second

A gateway is the natural place for:

- API key authentication
- request validation
- rate limiting
- audit logging

It protects the inference path seen by users or external systems.

### 3. SPIFFE/SPIRE later

SPIFFE/SPIRE is a more advanced step.
It is excellent for real platform security, but it is harder to learn because it introduces:

- workload identity
- certificates and trust bundles
- service-to-service authentication
- rotation and attestation concepts

That is worth doing, but only after the platform already has basic guardrails.

## What This Branch Adds

This branch starts with `Kyverno` foundations:

- beginner-friendly documentation
- two safe starter policies in `Audit` mode
- test manifests so you can see policy results without breaking the cluster
- a small KServe manifest hardening change so the downloader sidecar also declares resources

## Folder Guide

- [kyverno/README.md](/C:/Users/ademh/Desktop/kubeflowing/kubeflowingg/security/kyverno/README.md)
- [disallow-privileged-kserve.yaml](/C:/Users/ademh/Desktop/kubeflowing/kubeflowingg/security/kyverno/policies/disallow-privileged-kserve.yaml)
- [require-pod-resources-kserve.yaml](/C:/Users/ademh/Desktop/kubeflowing/kubeflowingg/security/kyverno/policies/require-pod-resources-kserve.yaml)
- [spire/README.md](/C:/Users/ademh/Desktop/kubeflowing/kubeflowingg/security/spire/README.md)

## Beginner Recommendation

Use the steps in [kyverno/README.md](/C:/Users/ademh/Desktop/kubeflowing/kubeflowingg/security/kyverno/README.md) first.

Do not start with `Enforce`.
Run everything in `Audit` mode first, confirm your existing workloads are compatible, then tighten policies one by one.
