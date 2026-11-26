# SPDX-License-Identifier: Apache-2.0
"""Durable request identity; result eviction never permits a second execution.

One database per owning API process/replica. Session-affine routing is required.
A crashed reservation is deliberately unrecoverable, not silently re-executed.
"""
import hashlib
import json
import os
import sqlite3
import time


class RequestConflict(ValueError):
    pass


class RequestLedger:
    def __init__(self, path=None, *, ttl=None, capacity=None):
        self.ttl = float(ttl if ttl is not None else os.getenv("VLLM_LUMINE_RESULT_TTL", "1800"))
        self.capacity = int(capacity or os.getenv("VLLM_LUMINE_REQUEST_CAPACITY", "100000"))
        self.db = sqlite3.connect(path or os.getenv("VLLM_LUMINE_REQUEST_DB", "/tmp/lumine-requests.sqlite3"))
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("CREATE TABLE IF NOT EXISTS sessions (sid TEXT PRIMARY KEY, retired INTEGER NOT NULL DEFAULT 0)")
        self.db.execute("CREATE TABLE IF NOT EXISTS requests (sid TEXT, rid TEXT, digest TEXT, status TEXT, result TEXT, updated REAL, PRIMARY KEY(sid,rid))")
        self.db.execute("CREATE INDEX IF NOT EXISTS result_expiry ON requests(status,updated)")
        self.active = set()
        self.stats = dict(misses=0, hits=0, conflicts=0, expired=0)

    def begin(self, sid, rid, body, *, initial=False, live=False):
        if not sid or not rid or len(sid) > 200 or len(rid) > 200:
            raise RequestConflict("Session ID and request ID required (max 200 chars)")
        digest = hashlib.sha256(json.dumps(body, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()
        now = time.time()
        with self.db:
            self.db.execute("UPDATE requests SET result=NULL, status='expired' WHERE status='done' AND updated < ?", (now-self.ttl,))
            row = self.db.execute("SELECT digest,status,result FROM requests WHERE sid=? AND rid=?", (sid,rid)).fetchone()
            if row:
                if row[0] != digest:
                    self.stats['conflicts'] += 1
                    raise RequestConflict("Request ID reused with different content")
                if row[1] == 'done':
                    if not live:
                        raise RequestConflict("Session lost; restore with a new session ID")
                    self.stats['hits'] += 1
                    return json.loads(row[2])
                if row[1] == 'pending' and (sid, rid) in self.active:
                    raise RequestConflict("Request in progress; retry the same request ID")
                self.stats['expired'] += 1
                raise RequestConflict("Result unavailable; restore with a new session ID")
            session = self.db.execute("SELECT retired FROM sessions WHERE sid=?", (sid,)).fetchone()
            if session and (session[0] or not live):
                raise RequestConflict("Session lost/retired; restore with a new session ID")
            if not session and not initial:
                raise RequestConflict("Unknown session; restore with a new ID and complete state")
            if self.db.execute("SELECT COUNT(*) FROM requests").fetchone()[0] >= self.capacity:
                raise RequestConflict("Request journal capacity reached; rotate journal only after retiring all sessions")
            self.db.execute("INSERT OR IGNORE INTO sessions(sid) VALUES(?)", (sid,))
            self.db.execute("INSERT INTO requests VALUES(?,?,?,'pending',NULL,?)", (sid,rid,digest,now))
        self.active.add((sid, rid))
        self.stats['misses'] += 1
        return None

    def finish(self, sid, rid, result):
        # Called only after the text/history transaction has completed. If a
        # crash intervenes, pending is an explicit failure on retry.
        with self.db:
            self.db.execute("UPDATE requests SET status='done',result=?,updated=? WHERE sid=? AND rid=?", (json.dumps(result,ensure_ascii=False),time.time(),sid,rid))

    def fail(self, sid, rid):
        self.active.discard((sid, rid))
        with self.db:
            # A disconnect after DONE must not invalidate an already committed result.
            self.db.execute("UPDATE requests SET status='failed',result=NULL WHERE sid=? AND rid=? AND status='pending'", (sid,rid))

    def retire(self, sid):
        with self.db:
            self.db.execute("UPDATE sessions SET retired=1 WHERE sid=?", (sid,))
            self.db.execute("UPDATE requests SET result=NULL,status='expired' WHERE sid=?", (sid,))

    def snapshot(self):
        return {**self.stats, 'records': self.db.execute('SELECT COUNT(*) FROM requests').fetchone()[0],
                'capacity': self.capacity, 'result_ttl_s': self.ttl}
