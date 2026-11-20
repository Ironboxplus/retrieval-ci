"""Robustness: the atomic conditional-UPDATE debit must never oversell under
concurrency. Hammers a real file-backed SQLite (WAL + busy_timeout) with 2N
concurrent 1-credit debits against a balance of N — exactly N must succeed, N
must hit InsufficientCredits, final balance 0, and no DB-lock errors."""
import threading
from concurrent.futures import ThreadPoolExecutor

from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker

from gateway.billing import ledger
from gateway.db.models import Base, Tier, User


def test_concurrent_debits_never_oversell(tmp_path):
    url = f"sqlite:///{(tmp_path / 'gw.db').as_posix()}"
    engine = create_engine(url, connect_args={"check_same_thread": False, "timeout": 30})

    @event.listens_for(engine, "connect")
    def _pragmas(dbapi_conn, _rec):
        cur = dbapi_conn.cursor()
        cur.execute("PRAGMA journal_mode=WAL")
        cur.execute("PRAGMA busy_timeout=30000")
        cur.close()

    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine, expire_on_commit=False)

    N = 25
    s = Session()
    u = User(oidc_sub="c", username="u", tier=Tier.paying, balance=N)
    s.add(u)
    s.commit()
    uid = u.id
    s.close()

    ok, broke, errors = [], [], []
    lock = threading.Lock()

    def worker(_i):
        s = Session()
        try:
            ledger.debit(s, uid, 1)
            s.commit()
            with lock:
                ok.append(1)
        except ledger.InsufficientCredits:
            s.rollback()
            with lock:
                broke.append(1)
        except Exception as exc:  # DB-lock or anything else => robustness failure
            s.rollback()
            with lock:
                errors.append(repr(exc))
        finally:
            s.close()

    with ThreadPoolExecutor(max_workers=12) as ex:
        list(ex.map(worker, range(2 * N)))

    s = Session()
    final = s.get(User, uid).balance
    s.close()

    assert errors == [], f"unexpected errors: {errors[:3]}"
    assert len(ok) == N            # exactly the funded amount succeeded
    assert len(broke) == N         # the rest were correctly refused
    assert final == 0              # no oversell, no double-count
