"""Bypass hung WMI service on Windows + Python 3.13.

Python 3.13 changed platform.py to use WMI queries (_wmi.exec_query) for
platform.machine(), platform.uname(), platform.win32_ver().  If the WMI
service is unresponsive (common after force-killing GPU processes), ANY
library that calls these functions — torch, numpy, etc. — deadlocks.

Import this module BEFORE anything that touches `platform`:

    import core.wmi_fix  # noqa: F401 — must be first
    import torch  # now safe
"""
import os
import platform
import sys

if sys.platform == "win32" and sys.version_info >= (3, 13):
    _TIMEOUT = float(os.environ.get("WMI_PROBE_TIMEOUT", "2"))

    def _probe_wmi() -> bool:
        """Return True if WMI responds within timeout, False if hung."""
        import threading

        result = [False]

        def _try():
            try:
                platform._wmi_query("Version", "Win32_OperatingSystem")
                result[0] = True
            except Exception:
                pass

        t = threading.Thread(target=_try, daemon=True)
        t.start()
        t.join(timeout=_TIMEOUT)
        return result[0]

    if not _probe_wmi():
        _cached_uname = platform.uname_result(
            "Windows",
            os.environ.get("COMPUTERNAME", "DESKTOP"),
            "10",
            "10.0.26200",
            "AMD64",
        )

        platform._wmi_query = lambda *a, **k: []
        platform.uname = lambda: _cached_uname
        platform.machine = lambda: "AMD64"
        platform.win32_ver = lambda *a, **k: (
            "10", "10.0.26200", "", "Multiprocessor Free"
        )

        import logging
        logging.getLogger(__name__).warning(
            "WMI service unresponsive (>%.1fs) — platform functions patched. "
            "Run 'net stop winmgmt && net start winmgmt' as admin to fix.",
            _TIMEOUT,
        )
