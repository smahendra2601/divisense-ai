"""Tier 1 — Data Acquisition: corporate-actions source interface + CSV impl.

Defines the abstract ``CorporateActionsSource`` with
``get_actions(ticker)``. MVP implementation (``CSVSource``) reads
``data/corporate_actions.csv``. The interface exists so a future
``NSEScraperSource`` can slot in without touching any other module.
"""
