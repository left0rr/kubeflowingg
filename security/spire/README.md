# SPIFFE / SPIRE Roadmap

`SPIFFE` and `SPIRE` are the right long-term direction for workload identity, but they are not the first security control to add to this project.

## What They Solve

They replace static identity assumptions with workload identity.

Instead of saying:

- "this pod is trusted because it is in namespace X"
- "this service knows the other service because of a hardcoded secret"

you move toward:

- each workload gets a cryptographic identity
- identities are short-lived and rotated
- services verify each other before communicating

## Why Not First

SPIFFE/SPIRE is powerful, but it adds several new concepts at once:

- trust domain
- workload attestation
- SVID certificates
- federation and identity validation

If introduced too early, it can feel like complexity without a visible payoff.

## Recommended Path For This Project

### Stage 1

Start with `Kyverno`:

- no privileged containers
- required resources
- later, approved image registries

### Stage 2

Add a `FastAPI` inference gateway:

- API keys
- request validation
- rate limiting
- audit logging

### Stage 3

Introduce `SPIFFE/SPIRE` for workload identity between:

- inference gateway
- KServe inference service
- future retraining trigger service

## What A First SPIFFE/SPIRE Goal Could Look Like

The first realistic goal is not "secure everything."
It is:

- deploy SPIRE server and agents in the local cluster
- issue identities to one or two workloads
- verify that one workload can authenticate another using SPIFFE identity

That would be a strong learning milestone.

## Good First Learning Questions

Before implementation, make sure you can answer:

1. What is the trust domain?
2. Which workloads actually need machine identity first?
3. Which current secrets or trust assumptions would SPIFFE replace?
4. How will you prove it is working in a demo?

## Recommendation

Treat SPIFFE/SPIRE as the advanced follow-up after Kyverno and the inference gateway are already in place.
That will make the project easier to explain and much easier to debug.
