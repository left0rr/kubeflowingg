# Kyverno Starter Guide

This guide explains the first practical security step for the project.

`Kyverno` is a Kubernetes policy engine. It watches objects like Pods and validates whether they follow your rules.

For this project, we start with two simple rules:

1. Pods in the `kserve` namespace should not run privileged containers.
2. Pods in the `kserve` namespace should declare CPU and memory requests/limits.

Both policies start in `Audit` mode so they report violations without blocking workloads.

## What Audit Mode Means

- `Audit`: show policy violations, but still allow the object
- `Enforce`: reject objects that violate the policy

As a beginner, always start with `Audit`.

## Files In This Folder

Policies:

- [disallow-privileged-kserve.yaml](/C:/Users/ademh/Desktop/kubeflowing/kubeflowingg/security/kyverno/policies/disallow-privileged-kserve.yaml)
- [require-pod-resources-kserve.yaml](/C:/Users/ademh/Desktop/kubeflowing/kubeflowingg/security/kyverno/policies/require-pod-resources-kserve.yaml)

Examples:

- [privileged-pod.yaml](/C:/Users/ademh/Desktop/kubeflowing/kubeflowingg/security/kyverno/examples/privileged-pod.yaml)
- [missing-resources-pod.yaml](/C:/Users/ademh/Desktop/kubeflowing/kubeflowingg/security/kyverno/examples/missing-resources-pod.yaml)

## Step 1. Install Kyverno

If you do not already have Kyverno:

```bash
kubectl create namespace kyverno
kubectl apply -f https://github.com/kyverno/kyverno/releases/latest/download/install.yaml
kubectl get pods -n kyverno
```

Wait until the Kyverno pods are `Running`.

## Step 2. Apply The Policies

```bash
kubectl apply -f security/kyverno/policies/disallow-privileged-kserve.yaml
kubectl apply -f security/kyverno/policies/require-pod-resources-kserve.yaml
```

Check them:

```bash
kubectl get cpol
kubectl describe cpol disallow-privileged-kserve
kubectl describe cpol require-pod-resources-kserve
```

## Step 3. Test The Policies Safely

Create a test Pod that is intentionally bad:

```bash
kubectl apply -f security/kyverno/examples/privileged-pod.yaml
kubectl apply -f security/kyverno/examples/missing-resources-pod.yaml
```

Because the policies are in `Audit` mode, these Pods should still be created.
The important part is that Kyverno reports violations.

Check policy reports:

```bash
kubectl get policyreport -A
kubectl get clusterpolicyreport
kubectl describe clusterpolicyreport
```

If your cluster does not show those objects, you can also inspect the Kyverno logs:

```bash
kubectl logs -n kyverno deploy/kyverno-admission-controller --tail=200
```

## Step 4. Check Your Real KServe Workload

Re-apply your inference service after these policies are present:

```bash
kubectl apply -f deployment/kserve/inference_service.yaml
```

Then inspect the generated Pods:

```bash
kubectl get pods -n kserve
kubectl describe pod -n kserve <pod-name>
```

This branch already updates [inference_service.yaml](/C:/Users/ademh/Desktop/kubeflowing/kubeflowingg/deployment/kserve/inference_service.yaml) so the `model-downloader` sidecar declares resource requests and limits too.

## Step 5. Move To Enforce Later

Only switch to `Enforce` after:

1. your real workloads pass
2. you understand the violations
3. you are confident the rules do not break system components

When you are ready, change this field in the policy manifests:

```yaml
validationFailureAction: Audit
```

to:

```yaml
validationFailureAction: Enforce
```

Then apply the policy again.

## Why We Are Not Starting With Image Allow Lists Yet

Image allow lists are useful, but easy to get wrong in a Kubeflow/KServe environment because controllers and generated Pods may use more images than expected.

That should come later after you inventory the real images used by:

- KServe
- Kubeflow Pipelines
- your inference sidecars
- supporting controllers

## Next Security Step After Kyverno

After you are comfortable with Kyverno, the next implementation step should be a `FastAPI` inference gateway in front of KServe.

That will give you:

- API key authentication
- request validation
- rate limiting
- request logging

Then, once the platform behavior is stable, move to `SPIFFE/SPIRE`.
