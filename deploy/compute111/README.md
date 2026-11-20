# Retrieval compute `.111` compatibility forward

After Retrieval has been fully restored and verified on `10.100.100.88`, this
user service keeps the trusted-LAN compatibility address
`10.100.100.111:8000` available by forwarding it to `.88:18080`. The SSH
destination is `.88`'s LAN address (not `127.0.0.1`) so the backend preserves
the trusted internal-door identity semantics for the legacy endpoint. It is not a
reverse proxy and must not be used for the public/BWH door.

The service deliberately has no credentials. Before installation, create a
dedicated `ed25519` key at
`~/.ssh/retrieval-forward-88_ed25519` on `.111`, record the `.88` host key in
`~/.ssh/known_hosts`, and append only its public key to `.88` user `arc`'s
`authorized_keys` with these restrictions:

```text
restrict,port-forwarding,permitopen="10.100.100.88:18080" <public-key>
```

Those restrictions permit this key to create only the one local forward to the
`.88` Retrieval internal door. They do not permit shell, agent, X11, or
arbitrary destination forwarding.

Install and verify only after `.111`'s old Retrieval backend has stopped and
freed port `8000`:

```bash
install -D -m 0644 retrieval-forward-88.service \
  ~/.config/systemd/user/retrieval-forward-88.service
systemctl --user daemon-reload
systemd-analyze --user verify ~/.config/systemd/user/retrieval-forward-88.service
systemctl --user enable --now retrieval-forward-88.service
curl --fail http://127.0.0.1:8000/api/stats
```

`ExitOnForwardFailure=yes` keeps the unit down rather than appearing healthy if
the source port is occupied or the target policy rejects the forward. `Linger`
must remain enabled for user `arc` on `.111` so this compatibility endpoint
survives logout/reboot.
