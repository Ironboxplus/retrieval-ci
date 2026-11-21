from pathlib import Path


ROOT = Path(__file__).resolve().parents[1] / "deploy" / "compute88"


def test_artifact_path_is_a_local_bind_mount_with_no_network_dependency():
    text = (ROOT / "mnt-lsb.mount").read_text(encoding="utf-8")
    assert "What=/home/arc/retrieval-storage" in text
    assert "Where=/mnt/lsb" in text
    assert "Type=none" in text
    assert "Options=bind" in text
    assert "network-online.target" not in text
    assert "credentials=" not in text


def test_old_nas_remains_at_a_separate_optional_mount():
    text = (ROOT / "mnt-retrieval_nas.mount").read_text(encoding="utf-8")
    assert "What=//10.100.100.9/lsb" in text
    assert "Where=/mnt/retrieval_nas" in text
    assert "credentials=/etc/smbcredentials224" in text
    assert "password=" not in text.lower()


def test_retrieval_smb_share_is_private_and_serves_only_local_project_storage():
    text = (ROOT / "retrieval-smb.conf").read_text(encoding="utf-8")
    assert "path = /home/arc/retrieval-storage" in text
    assert "guest ok = no" in text
    assert "map to guest = Never" in text
    assert "bind interfaces only = yes" in text
    assert "interfaces = lo 10.100.100.88/24" in text
    assert "force user = arc" in text
    assert "wide links = no" in text
    assert "smb ports = 445" in text
    assert "password=" not in text.lower()


def test_medical_and_paper_legacy_paths_stay_mounted_before_backend_start():
    text = (ROOT / "arc-retrival-backend.service").read_text(encoding="utf-8")
    assert "RequiresMountsFor=/mnt/lsb/retrival" in text
    assert "MEDICAL_DATA_ROOT=/mnt/lsb/retrival/medical" in text
