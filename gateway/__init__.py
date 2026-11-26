"""Gateway — the public app-tier (auth + billing + reverse-proxy).

Runs on the public web backend (BWH), in FRONT of the 4090 compute service. It owns
all sensitive state (users, sk-key hashes, balances, ledger, payment secrets); the
shared 4090 box never sees any of it. The 4090 stays pure compute behind an internal
token. See plan.md「付费多用户三层架构」.
"""
