
import os
import time

import requests

SEARCH_URL = "https://en.wikipedia.org/w/api.php"
SUMMARY_URL = "https://en.wikipedia.org/api/rest_v1/page/summary/"
HEADERS = {"User-Agent": "FactCheckerAgent/1.0 (educational project)"}


def _env_flag(name: str) -> bool:
    return os.getenv(name, "0").strip().lower() in ("1", "true", "yes")


def _do_search(question: str, timeout: float):

    params = {
        "action": "query",
        "list": "search",
        "srsearch": question,
        "format": "json",
    }

    response = requests.get(SEARCH_URL, params=params, headers=HEADERS, timeout=timeout)
    response.raise_for_status()
    data = response.json()

    search_results = data.get("query", {}).get("search", [])

    if not search_results:
        return []

    candidates = []

    for result in search_results[:3]:
        title = result["title"]
        summary_url = SUMMARY_URL + requests.utils.quote(title)

        summary_response = requests.get(summary_url, headers=HEADERS, timeout=timeout)
        summary_response.raise_for_status()
        summary_data = summary_response.json()

        candidates.append(
            {
                "title": summary_data.get("title"),
                "content": summary_data.get("extract"),
                "url": summary_data.get("content_urls", {}).get("desktop", {}).get("page"),
            }
        )

    return candidates


def search_wikipedia(
    question: str,
    simulate_failure: bool = False,
    timeout: float = 5.0,
    max_retries: int = 1,
):
    """
    Query Wikipedia for a question.

    simulate_failure: when True, skips the real network call entirely and
    forces a timeout-shaped failure. This exists so the agent's degradation
    path can be demonstrated on demand (toggled from the frontend or via
    SIMULATE_WIKI_FAILURE=1), instead of relying on Wikipedia actually being
    down during a recording. If simulate_failure is not passed explicitly,
    the SIMULATE_WIKI_FAILURE env var is used as the default, so this also
    still works exactly as described in the assessment brief when driven
    from the command line.

    Retries once (by default) on timeout only - a timeout is often
    transient, whereas a 4xx/JSON error is not going to fix itself on a
    second try within the same request, so we don't waste time retrying it.
    """

    if simulate_failure or (simulate_failure is False and _env_flag("SIMULATE_WIKI_FAILURE")):
        return {
            "source": "wikipedia",
            "status": "timeout",
            "content": None,
            "error": "Simulated failure (SIMULATE_WIKI_FAILURE / simulate_failure=True)",
        }

    attempts = max_retries + 1
    last_error = None

    for attempt in range(1, attempts + 1):
        try:
            candidates = _do_search(question, timeout)

            if not candidates:
                return {
                    "source": "wikipedia",
                    "status": "no_results",
                    "content": None,
                }

            return {
                "source": "wikipedia",
                "status": "success",
                "candidates": candidates,
                "retries_used": attempt - 1,
            }

        except requests.Timeout as e:
            last_error = str(e)
            if attempt < attempts:
                time.sleep(0.5 * attempt)  
                continue
            return {
                "source": "wikipedia",
                "status": "timeout",
                "content": None,
                "error": "Wikipedia request timed out" + (f" after {attempts} attempts" if max_retries else ""),
            }

        except requests.RequestException as e:
          
            return {
                "source": "wikipedia",
                "status": "error",
                "content": None,
                "error": str(e),
            }

        except ValueError:
            return {
                "source": "wikipedia",
                "status": "bad_response",
                "content": None,
                "error": "Wikipedia returned invalid JSON",
            }

    return {
        "source": "wikipedia",
        "status": "error",
        "content": None,
        "error": last_error or "Unknown error",
    }


