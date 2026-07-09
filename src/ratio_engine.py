"""Tier 2 — Intelligence: deterministic financial-ratio computation.

Pure pandas, NO LLM involvement — all financial numbers are computed in
code. Computes: payout ratio (5yr), dividend CAGR, FCF/dividend
coverage, consecutive-increase streak, current yield, debt/equity
trend, dividend consistency score, and recent dividend trajectory
(last 4 payouts: rising/flat/falling — powers "will it increase?"
answers). Returns a structured ``metrics`` dict.
"""
