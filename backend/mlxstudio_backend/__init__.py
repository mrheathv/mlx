"""Local MLX inference backend for MLX Studio.

Talks newline-delimited JSON over stdin/stdout to the Swift app — see
ARCHITECTURE.md at the repo root for why this process boundary exists and
how the wire protocol is shaped. Nothing in this package makes network
calls other than the one-time Hugging Face model download.
"""
