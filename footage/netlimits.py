"""Time limits and retries for every Google request the engine makes (added in 0.1.2, 2026-11-18).

- Each request gets at most REQUEST_TIMEOUT_S seconds, and never more than what is left of the whole
  estimate's budget (set by estimate() with start()).
- A Google 5xx, 429, timeout or dropped connection is retried once. If it still fails, GoogleUnavailable
  is raised so the caller reports google_unavailable (or timeout) instead of a false "no data".
- A Google 4xx (for example 404 "no roof data here") is returned to the caller as before, untouched.
"""
import socket, time, urllib.error, urllib.request

REQUEST_TIMEOUT_S = 10.0
RETRY_PAUSE_S = 1.0
_deadline = None          # wall clock time the whole estimate must finish by (one estimate at a time)


class GoogleUnavailable(Exception):
    """Google did not answer in time or answered with a server error, twice."""
    def __init__(self, message, timed_out=False):
        super().__init__(message)
        self.timed_out = timed_out


def start(budget_s):
    global _deadline
    _deadline = time.time() + budget_s if budget_s else None


def stop():
    global _deadline
    _deadline = None


def remaining():
    return None if _deadline is None else _deadline - time.time()


def past_deadline():
    r = remaining()
    return r is not None and r <= 0


def _timeout():
    r = remaining()
    if r is None:
        return REQUEST_TIMEOUT_S
    if r <= 0.2:
        raise GoogleUnavailable("estimate time limit reached", timed_out=True)
    return min(REQUEST_TIMEOUT_S, r)


def read(url):
    """GET url and return the body bytes. Raises urllib.error.HTTPError for 4xx, GoogleUnavailable for
    5xx/429/timeouts after one retry."""
    last = None
    for attempt in range(2):
        try:
            with urllib.request.urlopen(url, timeout=_timeout()) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code < 500 and e.code != 429:
                raise
            last = "Google answered HTTP %s" % e.code
        except (socket.timeout, TimeoutError) as e:
            last = "Google did not answer in time"
        except urllib.error.URLError as e:
            last = "Google connection failed (%s)" % type(getattr(e, "reason", e)).__name__
        except (ConnectionError, OSError) as e:
            last = "Google connection failed (%s)" % type(e).__name__
        if attempt == 0:
            r = remaining()
            if r is not None and r <= RETRY_PAUSE_S + 0.5:
                break
            time.sleep(RETRY_PAUSE_S)
    raise GoogleUnavailable(last or "Google unavailable", timed_out=past_deadline())
