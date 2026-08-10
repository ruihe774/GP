"""Component B: the realtime commentary agent.

Consumes Component A's event stream, decides when there is something worth
saying, and says it. The decision is the hard part and lives in `policy.py`;
everything else here is plumbing around it.
"""
